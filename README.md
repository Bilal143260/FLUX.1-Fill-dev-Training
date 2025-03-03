# FLUX.1-Fill-dev-Training

This repository contains the code for training FLUX.1-Fill-dev, a powerful image inpainting model. This varient is specifically trained for high-quality background changing and generation in images. This model excels at:

- **Seamless background replacement** for portraits, product photos, and landscapes
- **Context-aware background generation** that matches the foreground subject
- **Realistic background transitions** without edge artifacts or color inconsistencies
- **Diverse background styles** including solid colors, gradients, natural scenes, and interior environments

The training approach focuses on maintaining foreground subject integrity while completely transforming backgrounds, making it ideal for:

- Portrait photography enhancement
- E-commerce product image standardization
- Real estate virtual staging
- Social media content creation

Unlike traditional image matting or background removal tools, FLUX.1-Fill-dev uses advanced diffusion techniques to generate photorealistic backgrounds that complement the subject, ensuring natural lighting, perspective, and shadows.

## Setup and Installation

### Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/FLUX.1-Fill-dev-Training.git
cd FLUX.1-Fill-dev-Training

# Install dependencies
pip install accelerate transformers diffusers wandb
```

### Environment Setup

Make sure you have access to the following:
- A GPU with 80GB VRAM (H100 recommended)
- Required datasets: "raresense/BGData" and "raresense/BGData_Validation" available on Hugging Face

## Training Instructions

### Configuration

The main training script `train_model.sh` contains key variables that you can modify:

```bash
# Main data columns
export SOURCE_COLUMN="ghost_images"  # Column containing source images
export TARGET_COLUMN="target"        # Column containing target images
export MASK_COLUMN="binary_mask"     # Column containing binary masks
export CAPTION_COLUMN="prompt"       # Column containing text prompts

# Model and dataset configuration
export MODEL_NAME="black-forest-labs/FLUX.1-Fill-dev"
export TRAIN_DATASET_NAME="raresense/BGData"
export TEST_DATASET_NAME="raresense/BGData_Validation"
export OUTPUT_DIR="trained-flux-inpaint"
```

### Running Training

To start training:

```bash
# Make sure the script is executable
chmod +x train_model.sh

# Run the training script
./train_model.sh
```

### Resume Training

To resume training from a checkpoint, uncomment the following line in `train_model.sh`:

```bash
# --resume_from_checkpoint="latest"  \
```

## Training Results

Below are examples of the model's performance before and after fine-tuning.

### Example 1:

| Source Image | Mask | Generated Output | Prompt |
|:------------:|:----:|:---------------------:|:----------------:|
| ![Source](https://cdn.discordapp.com/attachments/961218697094123561/1346079455235280896/55.jpg?ex=67c6e1af&is=67c5902f&hm=beadbb7ee6128abdaa5cf7ad1ce2d9bbb9ad2f7e7742aaffd0e4cf018d76de42&) | ![Mask](https://cdn.discordapp.com/attachments/961218697094123561/1346079455596249158/image_15.png?ex=67c6e1af&is=67c5902f&hm=dfd436581e2c54a01c70362f2dbdc5f8a60b0beba3f8ba2a9a96201bf96789d3&) | ![Output](https://cdn.discordapp.com/attachments/961218697094123561/1346079456002838548/image_17.png?ex=67c6e1af&is=67c5902f&hm=86987ef1d33766d98f10b4e6866813f631c4f716563f16799b36e8fcf06eed93&) | A model posing in front of a grand marble staircase inside a luxury mansion, opulent lighting, chandeliers glistening, cinematic ambiance, ultra-realistic, Vogue editorial, 8K |

### Example 2: 

| Source Image | Mask | Target (Ground Truth) | Generated Output |
|:------------:|:----:|:---------------------:|:----------------:|
| ![Source](path/to/source2.jpg) | ![Mask](path/to/mask2.jpg) | ![Target](path/to/target2.jpg) | ![Output](path/to/output2.jpg) |

## Model Inference

After training, you can use the model for inference:

```python
from diffusers import FluxFillPipeline
import torch

# Load the fine-tuned model
pipeline = FluxFillPipeline.from_pretrained(
    "path/to/trained-flux-inpaint",
    torch_dtype=torch.float16
).to("cuda")

# Run inference
output = pipeline(
    prompt="your text prompt here",
    image=source_image,
    mask_image=mask_image,
    num_inference_steps=28,
    guidance_scale=30
).images[0]

# Save the result
output.save("output.png")
```

## Monitoring

This training script uses Weights & Biases for monitoring. Make sure you're logged in to your W&B account:

```bash
wandb login
```

You can track the training progress, visualize validation images, and monitor loss in the W&B dashboard.
