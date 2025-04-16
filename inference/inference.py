"""
Inference module for ChimeraAI.

This module handles animation generation from trained models,
implementing the image animation pipeline with hybrid guidance.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import yaml
import logging
from tqdm import tqdm
import time

from models.dit import DiT, create_dit_model
from models.guidance import HybridGuidanceSystem, create_guidance_system
from utils.config import load_config
from utils.visualize import create_video, save_image_grid


class ChimeraAIInference:
    """
    Inference class for generating animations with ChimeraAI.
    
    This class provides methods for animating still images using
    motion guidance from facial, head, and body inputs.
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda"
    ):
        """
        Initialize the inference engine.
        
        Args:
            config_path: Path to the inference configuration file
            checkpoint_path: Path to the model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logger
        self.logger = logging.getLogger("ChimeraAIInference")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load models from checkpoint."""
        self.logger.info(f"Loading models from checkpoint: {self.checkpoint_path}")
        
        # Initialize models
        self.model = create_dit_model(self.config['model']).to(self.device)
        self.guidance_system = create_guidance_system(self.config['hybrid_guidance']).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model'])
        self.guidance_system.load_state_dict(checkpoint['guidance_system'])
        
        # Set to evaluation mode
        self.model.eval()
        self.guidance_system.eval()
        
        self.logger.info("Models loaded successfully")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess an input image for inference.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        target_size = self.config['input']['image_size']
        
        if self.config['input']['keep_aspect_ratio']:
            # Preserve aspect ratio
            h, w = image.shape[:2]
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            
            # Pad to target size
            pad_h = max(0, target_size - new_h)
            pad_w = max(0, target_size - new_w)
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            
            image = cv2.copyMakeBorder(
                image, top, bottom, left, right, 
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        else:
            # Resize without preserving aspect ratio
            image = cv2.resize(image, (target_size, target_size))
        
        # Convert to tensor
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        return image.to(self.device)
    
    def extract_motion_guidance(
        self,
        motion_source_path: str,
        num_frames: int = 73
    ) -> Dict[str, torch.Tensor]:
        """
        Extract motion guidance signals from a source video or sequence.
        
        Args:
            motion_source_path: Path to the motion source (video or directory)
            num_frames: Number of frames to extract
            
        Returns:
            Dictionary of motion guidance tensors
        """
        self.logger.info(f"Extracting motion guidance from: {motion_source_path}")
        
        # Check if source is a video or directory
        is_video = motion_source_path.endswith(('.mp4', '.avi', '.mov'))
        
        # Initialize storage for guidance signals
        facial_data = []
        head_spheres = []
        body_skeletons = []
        
        if is_video:
            # Extract frames from video
            cap = cv2.VideoCapture(motion_source_path)
            
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {motion_source_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Determine frame indices to extract
            if num_frames >= total_frames:
                frame_indices = list(range(total_frames))
            else:
                # Sample frames evenly
                frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            
            for idx in tqdm(frame_indices, desc="Extracting frames"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process frame to extract guidance signals
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract facial data (simplified placeholder)
                face_tensor = self._extract_facial_data(frame_rgb)
                facial_data.append(face_tensor)
                
                # Extract head sphere (simplified placeholder)
                head_sphere = self._extract_head_sphere(frame_rgb)
                head_spheres.append(head_sphere)
                
                # Extract body skeleton (simplified placeholder)
                body_skeleton = self._extract_body_skeleton(frame_rgb)
                body_skeletons.append(body_skeleton)
            
            cap.release()
            
        else:
            # Extract from image sequence directory
            image_files = sorted([
                f for f in os.listdir(motion_source_path) 
                if f.endswith(('.jpg', '.jpeg', '.png'))
            ])
            
            if not image_files:
                raise ValueError(f"No images found in directory: {motion_source_path}")
            
            # Limit to requested number of frames
            if len(image_files) > num_frames:
                # Sample frames evenly
                indices = [int(i * len(image_files) / num_frames) for i in range(num_frames)]
                image_files = [image_files[i] for i in indices]
            
            for img_file in tqdm(image_files, desc="Processing images"):
                img_path = os.path.join(motion_source_path, img_file)
                frame = cv2.imread(img_path)
                
                if frame is None:
                    self.logger.warning(f"Failed to load image: {img_path}")
                    continue
                
                # Process frame to extract guidance signals
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract facial data (simplified placeholder)
                face_tensor = self._extract_facial_data(frame_rgb)
                facial_data.append(face_tensor)
                
                # Extract head sphere (simplified placeholder)
                head_sphere = self._extract_head_sphere(frame_rgb)
                head_spheres.append(head_sphere)
                
                # Extract body skeleton (simplified placeholder)
                body_skeleton = self._extract_body_skeleton(frame_rgb)
                body_skeletons.append(body_skeleton)
        
        # Stack tensors
        facial_data = torch.stack(facial_data, dim=0).unsqueeze(0)  # [1, T, ...]
        head_spheres = torch.stack(head_spheres, dim=0).unsqueeze(0)  # [1, T, ...]
        body_skeletons = torch.stack(body_skeletons, dim=0).unsqueeze(0)  # [1, T, ...]
        
        return {
            'facial_data': facial_data,
            'head_spheres': head_spheres,
            'body_skeletons': body_skeletons,
            'num_frames': len(facial_data[0])
        }
    
    def _extract_facial_data(self, image: np.ndarray) -> torch.Tensor:
        """
        Extract facial data from an image.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Facial data tensor
        """
        # This is a placeholder implementation
        # In a real implementation, we would use proper facial landmark detection
        
        # For this example, we'll create a random tensor of the expected shape
        # In reality, this would use the same preprocessing as in training
        
        # Create a tensor of shape [3, 224, 224] (RGB facial image)
        image = cv2.resize(image, (224, 224))
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # [3, 224, 224]
        
        return tensor.to(self.device)
    
    def _extract_head_sphere(self, image: np.ndarray) -> torch.Tensor:
        """
        Extract 3D head sphere from an image.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Head sphere tensor
        """
        # This is a placeholder implementation
        # In a real implementation, we would use proper 3D face reconstruction
        
        # Get the resolution from config
        resolution = 64  # Default value
        if 'head_sphere' in self.config['motion_guidance']:
            resolution = self.config['motion_guidance']['head_sphere'].get('resolution', 64)
        
        # Create a tensor of shape [1, resolution, resolution, resolution]
        tensor = torch.zeros(1, resolution, resolution, resolution)
        
        # Fill in a sphere in the center (simplified)
        center = resolution // 2
        radius = resolution // 4
        
        # Create a simple sphere
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    dist = np.sqrt((i - center)**2 + (j - center)**2 + (k - center)**2)
                    if dist <= radius:
                        tensor[0, i, j, k] = 1.0
        
        return tensor.to(self.device)
    
    def _extract_body_skeleton(self, image: np.ndarray) -> torch.Tensor:
        """
        Extract body skeleton from an image.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Body skeleton tensor
        """
        # This is a placeholder implementation
        # In a real implementation, we would use proper pose estimation
        
        # Get number of keypoints from config
        num_keypoints = 18  # Default value
        if 'body_skeleton' in self.config['motion_guidance']:
            num_keypoints = self.config['motion_guidance']['body_skeleton'].get('keypoints', 18)
        
        # Create a tensor of shape [num_keypoints, 3]
        tensor = torch.zeros(num_keypoints, 3)
        
        # Set random positions for keypoints (simplified)
        h, w = image.shape[:2]
        for i in range(num_keypoints):
            tensor[i, 0] = torch.rand(1) * w  # x
            tensor[i, 1] = torch.rand(1) * h  # y
            tensor[i, 2] = torch.rand(1)  # confidence
        
        return tensor.to(self.device)
    
    def ddim_sample(
        self,
        reference_image: torch.Tensor,
        guidance_signals: Dict[str, torch.Tensor],
        frame_idx: int,
        guidance_scale: float = 7.5,
        num_steps: int = 50
    ) -> torch.Tensor:
        """
        Generate a frame using DDIM sampling with hybrid guidance.
        
        Args:
            reference_image: Reference image tensor [1, 3, H, W]
            guidance_signals: Dictionary of guidance tensors
            frame_idx: Index of the frame to generate
            guidance_scale: Scale for classifier-free guidance
            num_steps: Number of sampling steps
            
        Returns:
            Generated frame
        """
        # Extract guidance for the specific frame
        facial_data = guidance_signals['facial_data'][:, frame_idx:frame_idx+1] if 'facial_data' in guidance_signals else None
        head_spheres = guidance_signals['head_spheres'][:, frame_idx:frame_idx+1] if 'head_spheres' in guidance_signals else None
        body_skeletons = guidance_signals['body_skeletons'][:, frame_idx:frame_idx+1] if 'body_skeletons' in guidance_signals else None
        
        # Reference frames for appearance guidance (include the original reference)
        reference_frames = reference_image.unsqueeze(1)  # [1, 1, 3, H, W]
        
        # Generate guidance tokens
        with torch.no_grad():
            guidance_outputs = self.guidance_system(
                facial_images=facial_data,
                head_spheres=head_spheres,
                body_skeletons=body_skeletons,
                reference_frames=reference_frames
            )
        
        # Initialize with random noise
        x = torch.randn_like(reference_image)
        
        # Setup diffusion parameters
        beta_start = 0.0001
        beta_end = 0.02
        noise_steps = 1000
        betas = torch.linspace(beta_start, beta_end, noise_steps, device=self.device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        
        # Sampling steps
        timesteps = torch.linspace(noise_steps - 1, 0, num_steps, dtype=torch.long, device=self.device)
        
        # DDIM sampling loop
        for i, t in enumerate(tqdm(timesteps, desc=f"Generating frame {frame_idx}")):
            # Expand time step to batch dimension
            t_batch = t.expand(x.shape[0])
            
            # Predict noise
            with torch.no_grad():
                predicted_noise = self.model(
                    x,
                    t_batch,
                    facial_guidance=guidance_outputs.get('facial_tokens', None),
                    head_sphere_guidance=guidance_outputs.get('head_sphere_tokens', None),
                    body_skeleton_guidance=guidance_outputs.get('body_tokens', None)
                )
            
            # Apply classifier-free guidance
            if guidance_scale > 1.0:
                # In a full implementation, we would have a way to get unconditional output
                # For simplicity, we'll skip that part in this example
                # unconditional_output = model(x, t_batch, None, None, None)
                # predicted_noise = unconditional_output + guidance_scale * (predicted_noise - unconditional_output)
                pass
            
            # DDIM update step
            alpha = alphas_cumprod[t]
            alpha_prev = alphas_cumprod_prev[t]
            sigma = 0.0  # Can be tuned for different sampling behaviors
            
            # Compute predicted original sample
            pred_original_sample = (x - sqrt_one_minus_alphas_cumprod[t] * predicted_noise) / torch.sqrt(alpha)
            
            # Get direction pointing to next x
            direction = torch.sqrt(1 - alpha_prev - sigma ** 2) * predicted_noise
            
            # Update x
            x = torch.sqrt(alpha_prev) * pred_original_sample + direction
            
            # Optional: Add noise for stochasticity
            noise = sigma * torch.randn_like(x)
            x = x + noise
        
        # Normalize to [0, 1]
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        
        return x
    
    def generate_animation(
        self,
        reference_image_path: str,
        motion_source_path: str,
        output_path: str,
        num_frames: int = 73,
        guidance_scale: float = 2.5,
        num_steps: int = 50
    ) -> str:
        """
        Generate an animation from a reference image and motion source.
        
        Args:
            reference_image_path: Path to the reference image
            motion_source_path: Path to the motion source (video or directory)
            output_path: Path to save the output animation
            num_frames: Number of frames to generate
            guidance_scale: Scale for classifier-free guidance
            num_steps: Number of sampling steps per frame
            
        Returns:
            Path to the output animation
        """
        self.logger.info(f"Generating animation with {num_frames} frames")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Preprocess reference image
        reference_image = self.preprocess_image(reference_image_path)
        
        # Extract motion guidance signals
        guidance_signals = self.extract_motion_guidance(motion_source_path, num_frames)
        
        # Adjust num_frames if necessary
        num_frames = min(num_frames, guidance_signals['num_frames'])
        
        # Generate frames
        frames = []
        for i in range(num_frames):
            frame = self.ddim_sample(
                reference_image,
                guidance_signals,
                i,
                guidance_scale,
                num_steps
            )
            
            # Convert to numpy for saving
            frame_np = frame.squeeze().permute(1, 2, 0).cpu().numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
            
            frames.append(frame_np)
            
            # Optional: Save individual frame
            if self.config['output']['save_individual_frames']:
                frame_dir = os.path.splitext(output_path)[0] + "_frames"
                os.makedirs(frame_dir, exist_ok=True)
                frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
        
        # Create video from frames
        fps = 30  # Default value
        if 'fps' in self.config['animation']:
            fps = self.config['animation']['fps']
        
        # Save animation
        if output_path.endswith('.mp4'):
            create_video(frames, output_path, fps)
        elif output_path.endswith('.gif'):
            create_video(frames, output_path, fps, is_gif=True)
        else:
            # Default to mp4
            output_path = os.path.splitext(output_path)[0] + '.mp4'
            create_video(frames, output_path, fps)
        
        self.logger.info(f"Animation saved to: {output_path}")
        
        # Generate preview grid if requested
        if self.config['output'].get('save_preview_grid', True):
            preview_frames = frames[::max(1, num_frames // 9)][:9]  # Up to 9 frames for preview
            grid_path = os.path.splitext(output_path)[0] + '_preview.jpg'
            save_image_grid(preview_frames, grid_path, rows=3)
            self.logger.info(f"Preview grid saved to: {grid_path}")
        
        return output_path


def animate(
    config_path: str,
    checkpoint_path: str,
    reference_image_path: str,
    motion_source_path: str,
    output_path: str,
    device: str = "cuda"
):
    """
    Generate an animation using ChimeraAI.
    
    Args:
        config_path: Path to the inference configuration file
        checkpoint_path: Path to the model checkpoint
        reference_image_path: Path to the reference image
        motion_source_path: Path to the motion source (video or directory)
        output_path: Path to save the output animation
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        Path to the output animation
    """
    # Initialize the inference engine
    inference = ChimeraAIInference(config_path, checkpoint_path, device)
    
    # Generate animation
    return inference.generate_animation(
        reference_image_path,
        motion_source_path,
        output_path
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate animations with ChimeraAI")
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="Path to inference config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference image")
    parser.add_argument("--motion", type=str, required=True, help="Path to motion source (video or directory)")
    parser.add_argument("--output", type=str, required=True, help="Path to save output animation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    animate(
        args.config,
        args.checkpoint,
        args.reference,
        args.motion,
        args.output,
        args.device
    )
