import copy
import logging
import math
import os
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed, DistributedType
from tqdm.auto import tqdm
from tqdm import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast
from parser_helper import parse_args
from train_utils import  prepare_fill_with_mask, prepare_latents, encode_images_to_latents
from diffusers import FluxTransformer2DModel, FluxFillPipeline
from diffusers.image_processor import VaeImageProcessor
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module
from data_module import BgAiDataset, collate_fn
from metrics import calculate_fid_score
if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.2")

logger = get_logger(__name__)

def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two

def log_validation(
    pipeline,
    args,
    accelerator,
    epoch,
    dataloader,
    tag,
    is_final_validation=False,
):
    logger.info(f"Running {tag}... \n ")

    pipeline = pipeline.to(accelerator.device)
    # Use appropriate precision context
    if accelerator.mixed_precision == "bf16":
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    elif accelerator.mixed_precision == "fp16":
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    
    with autocast_ctx:
        images = []
        prompts = []
        control_images = []
        control_masks = []
        target_images = []
        
        # Process full dataset - no limit on samples for validation
        # This ensures we use all images from the test dataset
        
        for batch in dataloader:
            prompt = batch['caption']
            control_image = batch['source_image']
            control_mask = batch['mask']
            target_image = batch['target_image']
            
            # Process in smaller batches to avoid memory issues
            validation_batch_size = max(1, args.train_batch_size // 2)  # Reduce batch size for validation
            
            for i in range(0, len(prompt), validation_batch_size):
                batch_prompt = prompt[i:i+validation_batch_size]
                batch_control_image = control_image[i:i+validation_batch_size]
                batch_control_mask = control_mask[i:i+validation_batch_size]
                batch_target_image = target_image[i:i+validation_batch_size]
                
                result = pipeline(
                    prompt=batch_prompt,
                    height=args.height,
                    width=args.width,
                    image=batch_control_image, 
                    mask_image=batch_control_mask,
                    num_inference_steps=28,
                    generator=generator,
                    guidance_scale=30,
                ).images
                
                images.extend(result)
                prompts.extend(batch_prompt)
                control_images.extend(batch_control_image)
                control_masks.extend(batch_control_mask)
                target_images.extend(batch_target_image)

    # Calculate FID score if enabled
    fid_score = None
    if args.calculate_fid and len(images) > 0:
        logger.info("Calculating FID score...")
        try:
            # Use the appropriate precision mode for FID calculation
            fid_precision = getattr(args, 'fid_precision', args.mixed_precision if args.mixed_precision else 'fp16')
            
            fid_score = calculate_fid_score(
                generated_images=images,
                target_images=target_images,
                device=accelerator.device,
                precision=fid_precision,
                batch_size=getattr(args, 'fid_batch_size', 32),
                use_memory_management=True
            )
            
            logger.info(f"FID Score: {fid_score:.4f}")
            
        except Exception as e:
            logger.error(f"Error calculating FID score: {str(e)}")
            fid_score = None

    # Log to trackers
    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted_images = []
            
            for input_img, mask_img, gen_img, tgt_img, prompt in zip(control_images, control_masks, images, target_images, prompts):
                formatted_images.append(wandb.Image(input_img, caption="Source Image"))
                formatted_images.append(wandb.Image(mask_img, caption="Mask"))
                formatted_images.append(wandb.Image(tgt_img, caption="Target Image"))
                formatted_images.append(wandb.Image(gen_img, caption=prompt))
            
            log_dict = {tracker_key: formatted_images}
            
            # Add FID score to log if available
            if fid_score is not None:
                log_dict[f"{tracker_key}/fid_score"] = fid_score
            
            tracker.log(log_dict)
    
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return images

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    dtype = text_encoders[0].dtype
    device = device if device is not None else text_encoders[1].device
    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    
    vae_scale_factor = (
        2 ** (len(vae.config.block_out_channels) - 1) if vae is not None else 8
    )
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_resize=True, do_convert_rgb=True, do_normalize=True)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor,
        do_resize=True,
        do_convert_grayscale=True,
        do_normalize=False,
        do_binarize=True,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, revision=args.revision, variant=args.variant, subfolder="transformer"
    )
    


    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    grad_params = [
        "transformer_blocks.0.",
        "transformer_blocks.1.",
        "transformer_blocks.2.",
        "transformer_blocks.3.",
        "transformer_blocks.4.",
        "transformer_blocks.5.",
        "transformer_blocks.6.",
        "transformer_blocks.7.",
        "transformer_blocks.8.",
        "transformer_blocks.9.",
        "transformer_blocks.10.",
        "transformer_blocks.11.",
        "transformer_blocks.12.",
        "transformer_blocks.13.",
        "transformer_blocks.14.",
        "transformer_blocks.15.",
        "transformer_blocks.16.",
        "transformer_blocks.17.",
        "transformer_blocks.18.",
        "single_transformer_blocks.0.",
        "single_transformer_blocks.1.",
        "single_transformer_blocks.2.",
        "single_transformer_blocks.3.",
        "single_transformer_blocks.4.",
        "single_transformer_blocks.5.",
        "single_transformer_blocks.6.",
        "single_transformer_blocks.7.",
        "single_transformer_blocks.8.",
        "single_transformer_blocks.9.",
        "single_transformer_blocks.10.",
        "single_transformer_blocks.13.",
        "single_transformer_blocks.14.",
        "single_transformer_blocks.15.",
        "single_transformer_blocks.16.",
        "single_transformer_blocks.17.",
        "single_transformer_blocks.18.",
        "single_transformer_blocks.19.",
        "single_transformer_blocks.20.",
        "single_transformer_blocks.21.",
        "single_transformer_blocks.22.",
        "single_transformer_blocks.23.",
        "single_transformer_blocks.24.",
        "single_transformer_blocks.25.",
        "single_transformer_blocks.26.",
        "single_transformer_blocks.27.",
        "single_transformer_blocks.28.",
        "single_transformer_blocks.29.",
        "single_transformer_blocks.30.",
        "single_transformer_blocks.31.",
        "single_transformer_blocks.32.",
        "single_transformer_blocks.33.",
        "single_transformer_blocks.34.",
        "single_transformer_blocks.35.",
        "single_transformer_blocks.36.",
        "single_transformer_blocks.37.",
    ]

    if args.train_base_model:
        transformer.requires_grad_(False)  # Set all parameters to not require gradients by default
        
        for name, param in transformer.named_parameters():
            if any(grad_param in name for grad_param in grad_params):
                if ("attn" in name):
                    param.requires_grad = True
                    print(f"Enabling gradients for: {name}")
        
    else:
        transformer.requires_grad_(False)   


    # #you can train your own layers
    # for n, param in transformer.named_parameters():
    #     print(n)
    #     if 'single_transformer_blocks' in n:
    #         param.requires_grad = False
    #     elif 'transformer_blocks' in n and '1.attn' in n:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    
    print(sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000, 'transformer parameters')

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)


    if args.gradient_checkpointing:
        if args.train_base_model:
            transformer.enable_gradient_checkpointing()
        

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                # if isinstance(model, DeepSpeedEngine):
                    # For DeepSpeed models, we need to get the underlying model
                    # model = model.module
                if isinstance(unwrap_model(model), FluxTransformer2DModel):
                    unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"),max_shard_size="50GB")
                elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                    if isinstance(unwrap_model(model), CLIPTextModelWithProjection):
                        unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder"),max_shard_size="50GB")
                    else:
                        unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_2"),max_shard_size="50GB")
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()
                else:
                    print('no weights')

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if isinstance(unwrap_model(model), FluxTransformer2DModel):
                load_model = FluxTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
            elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                try:
                    load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder")
                    model(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                except Exception:
                    try:
                        load_model = T5EncoderModel.from_pretrained(input_dir, subfolder="text_encoder_2")
                        model(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                    except Exception:
                        raise ValueError(f"Couldn't load the model of type: ({type(model)}).")
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")

            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimization parameters
    if args.train_base_model:
        transformer_parameters_with_lr = {"params": transformer.parameters(), "lr": args.learning_rate}
   
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:
    train_dataset = BgAiDataset(
        dataset_name=args.train_dataset_name,
        source_column_name=args.source_column,
        mask_column_name=args.mask_column,
        target_column_name=args.target_column,
        caption_column_name=args.caption_column,
        size=(args.width, args.height),
    )
    
    validation_dataset = BgAiDataset(
        dataset_name=args.test_dataset_name,
        source_column_name=args.source_column,
        mask_column_name=args.mask_column,
        target_column_name=args.target_column,
        caption_column_name=args.caption_column,
        size=(args.width, args.height),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
    )
    
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
    )

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            text_ids = text_ids.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.

    # Handle class prompt for prior-preservation.


    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )


    if args.train_base_model:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "BGAI-flux-inpaint"
        accelerator.init_trackers(tracker_name, config=vars(args), init_kwargs={"wandb": {"settings": wandb.Settings(code_dir=".")}})
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    epoch = first_epoch


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        if args.train_base_model:
            transformer.train()
        
        for step, batch in enumerate(train_dataloader):
            if args.train_base_model:
                models_to_accumulate = [transformer]
            
            
            with accelerator.accumulate(models_to_accumulate):
                # vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
                batch_size = batch["target_image"].shape[0]
                pixel_values = batch["target_image"].to(dtype=vae.dtype)
                prompts = batch["caption"]

                control_mask = batch["mask"].to(dtype=vae.dtype)
                control_image = batch["source_image"].to(dtype=vae.dtype)
                
                
                # print("image_proj.shape", image_proj.shape)

                # encode batch prompts when custom prompts are provided for each image -
                prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                    prompts, text_encoders, tokenizers
                )
                
                inpaint_cond, _, _ = prepare_fill_with_mask(
                    image_processor=image_processor,
                    mask_processor=mask_processor,
                    vae=vae,
                    vae_scale_factor=vae_scale_factor,
                    image=control_image,
                    mask=control_mask,
                    width=args.width,
                    height=args.height,
                    batch_size=batch_size,
                    num_images_per_prompt=1,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                
                
                # TODO: controlnet dropout might cause instability, need to run more experiments
                if args.dropout_prob > 0:
                    dropout = torch.nn.Dropout(p=args.dropout_prob)
                    inpaint_cond = dropout(inpaint_cond)

                model_input = encode_images_to_latents(vae, pixel_values, weight_dtype, args.height, args.width)
                
                latent_image_ids = prepare_latents(
                    vae_scale_factor,
                    batch_size,
                    args.height,
                    args.width,
                    weight_dtype,
                    accelerator.device,
                )

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxFillPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )
                
                # handle guidance
                # guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                guidance = torch.full([1], args.guidance_scale, device=accelerator.device)
                guidance = guidance.expand(model_input.shape[0])
                
                # print("before concat packed_noisy_model_input.shape", packed_noisy_model_input.shape, "inpaint_cond.shape", inpaint_cond.shape)
                
                if inpaint_cond is not None:
                    packed_noisy_model_input = torch.cat([packed_noisy_model_input, inpaint_cond], dim=-1)
                
                # print("guidance", guidance, "pooled_prompt_embeds.shape", pooled_prompt_embeds.shape, "prompt_embeds.shape", prompt_embeds.shape)
                
                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                
                # print("model_pred.shape", model_pred.shape, "prompt_embeds.shape", prompt_embeds.shape, "packed_noisy_model_input.shape", packed_noisy_model_input.shape, "refnet_image.shape", refnet_image.shape)
                # upscaling height & width as discussed in https://github.com/huggingface/diffusers/pull/9257#discussion_r1731108042
                model_pred = FluxFillPipeline._unpack_latents(
                    model_pred,
                    height=args.height,
                    width=args.width,
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.train_base_model:
                        params_to_clip = (
                            transformer.parameters()
                        )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
            
                # Run validation every args.validation_steps steps (e.g., every 5 steps)
                if global_step % args.validation_steps == 0:
                    pipeline = FluxFillPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        transformer=accelerator.unwrap_model(transformer),
                        torch_dtype=weight_dtype,
                        vae=vae,
                        tokenizer=tokenizer_one,
                        tokenizer_2=tokenizer_two,
                        text_encoder=text_encoder_one,
                        text_encoder_2=text_encoder_two,
                    )
                    
                    log_validation(
                        pipeline=pipeline,
                        args=args,
                        accelerator=accelerator,
                        dataloader=validation_dataloader,
                        tag="validation",
                        epoch=epoch,
                    )

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)

        pipeline = FluxFillPipeline.from_pretrained(args.pretrained_model_name_or_path, transformer=transformer)

        # save the pipeline
        pipeline.save_pretrained(args.output_dir, max_shard_size="50GB")

        # Final inference
        # Load previous pipeline
        pipeline = FluxFillPipeline.from_pretrained(
            args.output_dir,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)