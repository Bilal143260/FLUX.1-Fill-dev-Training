"""
FID Score Calculation Module for FLUX.1-Fill Training

This module provides functionality to calculate Fréchet Inception Distance (FID) scores
between generated and target images during validation.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import numpy as np
import tempfile
import os
from PIL import Image
import logging
from typing import List, Tuple, Optional, Union
from contextlib import nullcontext

logger = logging.getLogger(__name__)


class FIDCalculator:
    """
    Calculates FID scores between generated and target images.
    
    This class handles the FID calculation process including:
    - Image preprocessing and conversion
    - Inception feature extraction
    - FID computation with proper memory management
    - Support for different precision modes
    """
    
    def __init__(self, device: str = "cuda", precision: str = "fp16"):
        """
        Initialize FID calculator.
        
        Args:
            device: Device to run calculations on ("cuda" or "cpu")
            precision: Precision mode ("fp16", "bf16", or "fp32")
        """
        self.device = device
        self.precision = precision
        
        # Initialize Inception model for feature extraction
        self.inception_model = InceptionV3(
            output_blocks=[InceptionV3.BLOCK_INDEX_BY_DIM[2048]],
            resize_input=False,
            normalize_input=False
        ).to(device)
        
        # Set precision context
        if precision == "bf16":
            self.autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        elif precision == "fp16":
            self.autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
        else:
            self.autocast_ctx = nullcontext()
            
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess_images(self, images: List[Union[Image.Image, torch.Tensor]]) -> torch.Tensor:
        """
        Preprocess images for FID calculation.
        
        Args:
            images: List of PIL Images or tensors
            
        Returns:
            Preprocessed tensor of shape (N, 3, 299, 299)
        """
        processed_images = []
        
        for img in images:
            if isinstance(img, torch.Tensor):
                # Convert tensor to PIL Image
                if img.dim() == 4:
                    img = img.squeeze(0)
                if img.shape[0] == 3:  # CHW format
                    img = img.permute(1, 2, 0)
                img = img.cpu().numpy()
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
            
            # Apply preprocessing
            processed_img = self.transform(img)
            processed_images.append(processed_img)
        
        return torch.stack(processed_images)
    
    def extract_features(self, images: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        """
        Extract features from images using Inception model.
        
        Args:
            images: Preprocessed images tensor
            batch_size: Batch size for processing
            
        Returns:
            Feature vectors as numpy array
        """
        self.inception_model.eval()
        features = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size].to(self.device)
                
                with self.autocast_ctx:
                    batch_features = self.inception_model(batch)[0]
                    
                # Flatten features
                batch_features = batch_features.squeeze(-1).squeeze(-1)
                features.append(batch_features.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def calculate_fid(
        self,
        generated_images: List[Union[Image.Image, torch.Tensor]],
        target_images: List[Union[Image.Image, torch.Tensor]],
        batch_size: int = 32
    ) -> float:
        """
        Calculate FID score between generated and target images.
        
        Args:
            generated_images: List of generated images
            target_images: List of target images
            batch_size: Batch size for processing
            
        Returns:
            FID score as float
        """
        if len(generated_images) != len(target_images):
            raise ValueError("Number of generated and target images must match")
        
        if len(generated_images) == 0:
            logger.warning("No images provided for FID calculation")
            return float('inf')
        
        logger.info(f"Calculating FID score for {len(generated_images)} image pairs")
        
        try:
            # Preprocess images
            gen_tensor = self.preprocess_images(generated_images)
            tgt_tensor = self.preprocess_images(target_images)
            
            # Extract features
            gen_features = self.extract_features(gen_tensor, batch_size)
            tgt_features = self.extract_features(tgt_tensor, batch_size)
            
            # Calculate FID score
            fid_value = self._calculate_fid_from_features(gen_features, tgt_features)
            
            logger.info(f"FID Score: {fid_value:.4f}")
            return fid_value
            
        except Exception as e:
            logger.error(f"Error calculating FID score: {str(e)}")
            return float('inf')
        
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _calculate_fid_from_features(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Calculate FID score from precomputed features.
        
        Args:
            features1: Features from first image set
            features2: Features from second image set
            
        Returns:
            FID score
        """
        # Calculate mean and covariance
        mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
        mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)
        
        # Calculate FID score
        fid_value = fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        
        return fid_value
    
    def calculate_fid_with_memory_management(
        self,
        generated_images: List[Union[Image.Image, torch.Tensor]],
        target_images: List[Union[Image.Image, torch.Tensor]],
        max_batch_size: int = 16,
        max_images_per_chunk: int = 100
    ) -> float:
        """
        Calculate FID score with memory management for large datasets.
        
        Args:
            generated_images: List of generated images
            target_images: List of target images
            max_batch_size: Maximum batch size for processing
            max_images_per_chunk: Maximum images to process at once
            
        Returns:
            FID score as float
        """
        total_images = len(generated_images)
        
        if total_images <= max_images_per_chunk:
            return self.calculate_fid(generated_images, target_images, max_batch_size)
        
        logger.info(f"Processing {total_images} images in chunks of {max_images_per_chunk}")
        
        # Process in chunks and accumulate features
        gen_features_chunks = []
        tgt_features_chunks = []
        
        for i in range(0, total_images, max_images_per_chunk):
            end_idx = min(i + max_images_per_chunk, total_images)
            
            gen_chunk = generated_images[i:end_idx]
            tgt_chunk = target_images[i:end_idx]
            
            logger.info(f"Processing chunk {i//max_images_per_chunk + 1} of {(total_images-1)//max_images_per_chunk + 1}")
            
            # Preprocess chunk
            gen_tensor = self.preprocess_images(gen_chunk)
            tgt_tensor = self.preprocess_images(tgt_chunk)
            
            # Extract features
            gen_features = self.extract_features(gen_tensor, max_batch_size)
            tgt_features = self.extract_features(tgt_tensor, max_batch_size)
            
            gen_features_chunks.append(gen_features)
            tgt_features_chunks.append(tgt_features)
            
            # Clean up memory after each chunk
            del gen_tensor, tgt_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all features
        all_gen_features = np.concatenate(gen_features_chunks, axis=0)
        all_tgt_features = np.concatenate(tgt_features_chunks, axis=0)
        
        # Calculate FID from all features
        fid_value = self._calculate_fid_from_features(all_gen_features, all_tgt_features)
        
        logger.info(f"Final FID Score: {fid_value:.4f}")
        return fid_value


def calculate_fid_score(
    generated_images: List[Union[Image.Image, torch.Tensor]],
    target_images: List[Union[Image.Image, torch.Tensor]],
    device: str = "cuda",
    precision: str = "fp16",
    batch_size: int = 32,
    use_memory_management: bool = True
) -> float:
    """
    Convenience function to calculate FID score.
    
    Args:
        generated_images: List of generated images
        target_images: List of target images
        device: Device to run calculations on
        precision: Precision mode ("fp16", "bf16", or "fp32")
        batch_size: Batch size for processing
        use_memory_management: Whether to use memory management for large datasets
        
    Returns:
        FID score as float
    """
    calculator = FIDCalculator(device=device, precision=precision)
    
    if use_memory_management:
        return calculator.calculate_fid_with_memory_management(
            generated_images, target_images, batch_size
        )
    else:
        return calculator.calculate_fid(generated_images, target_images, batch_size)