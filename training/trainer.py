"""
Trainer implementation for ChimeraAI.

This module implements the progressive training pipeline described in the paper,
with three distinct stages focusing on different aspects of the model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any
import yaml
import logging
import time
from pathlib import Path

from models.dit import DiT, create_dit_model
from models.guidance import HybridGuidanceSystem, create_guidance_system
from data.dataset import create_dataloader
from training.losses import ReconstructionLoss, PerceptualLoss, IdentityLoss, TemporalConsistencyLoss
from utils.logging import setup_logger, log_metrics, log_images


class ChimeraAITrainer:
    """
    Trainer for the ChimeraAI model.
    
    Implements the progressive training strategy described in the paper:
    Stage 1: Train with 3D body skeletons and head spheres
    Stage 2: Add implicit facial representations and train face motion encoder
    Stage 3: Unfreeze all parameters and train the full model
    """
    
    def __init__(
        self,
        config_path: str,
        resume_from: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to the training configuration file
            resume_from: Optional path to a checkpoint to resume training from
            device: Device to train on ('cuda' or 'cpu')
        """
        self.config_path = config_path
        self.resume_from = resume_from
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        self.logger = setup_logger("ChimeraAITrainer")
        self.logger.info(f"Initializing ChimeraAITrainer with config from {config_path}")
        
        # Setup models, optimizers, and losses
        self._setup_training()
        
        # Load checkpoint if provided
        if resume_from:
            self._load_checkpoint(resume_from)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_training(self):
        """Setup models, optimizers, schedules, and losses."""
        # Get training parameters
        self.seed = self.config['training']['seed']
        self.epochs = self.config['training']['epochs']
        self.fp16 = self.config['training']['fp16']
        self.gradient_clipping = self.config['training']['gradient_clipping']
        self.save_every = self.config['training']['save_every']
        self.eval_every = self.config['training']['eval_every']
        
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        
        # Initialize models
        self.logger.info("Initializing DiT model")
        self.model = create_dit_model(self.config['model']).to(self.device)
        
        self.logger.info("Initializing Guidance System")
        self.guidance_system = create_guidance_system(self.config['hybrid_guidance']).to(self.device)
        
        # Initialize optimizers
        self.optimizer = self._create_optimizer(
            list(self.model.parameters()) + list(self.guidance_system.parameters())
        )
        
        # Initialize learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler(self.optimizer)
        
        # Initialize losses
        self.losses = {
            'reconstruction': ReconstructionLoss().to(self.device),
            'perceptual': PerceptualLoss().to(self.device),
            'identity': IdentityLoss().to(self.device),
            'temporal': TemporalConsistencyLoss().to(self.device)
        }
        
        # Setup mixed precision training if enabled
        self.scaler = torch.cuda.amp.GradScaler() if self.fp16 else None
        
        # Initialize training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Progressive training state
        self.progressive_config = self.config['progressive_training']
        self.current_stage = 0
        self.stages = self.progressive_config['stages']
    
    def _create_optimizer(self, parameters) -> torch.optim.Optimizer:
        """Create optimizer from configuration."""
        optimizer_config = self.config['optimizer']
        optimizer_type = optimizer_config['type']
        
        if optimizer_type == 'adam':
            return optim.Adam(
                parameters,
                lr=self.config['training']['learning_rate'],
                betas=optimizer_config['betas']
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                parameters,
                lr=self.config['training']['learning_rate'],
                weight_decay=optimizer_config['weight_decay'],
                betas=optimizer_config['betas']
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                parameters,
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=optimizer_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _create_lr_scheduler(self, optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from configuration."""
        scheduler_type = self.config['training']['lr_scheduler']
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs
            )
        elif scheduler_type == 'linear':
            return optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.epochs
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.epochs // 3,
                gamma=0.1
            )
        else:
            return None
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model and guidance system
        self.model.load_state_dict(checkpoint['model'])
        self.guidance_system.load_state_dict(checkpoint['guidance_system'])
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.lr_scheduler and 'lr_scheduler' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.current_stage = checkpoint['current_stage']
        
        # Load scaler if using mixed precision
        if self.fp16 and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
    
    def _save_checkpoint(self, path: str, is_best: bool = False):
        """Save checkpoint during training."""
        checkpoint = {
            'model': self.model.state_dict(),
            'guidance_system': self.guidance_system.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'current_stage': self.current_stage,
            'config': self.config
        }
        
        if self.lr_scheduler:
            checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        if self.fp16:
            checkpoint['scaler'] = self.scaler.state_dict()
        
        # Save the checkpoint
        torch.save(checkpoint, path)
        
        # If this is the best model, create a copy
        if is_best:
            best_path = os.path.join(os.path.dirname(path), 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
    
    def _setup_stage(self, stage_idx: int):
        """Setup a specific training stage."""
        if stage_idx >= len(self.stages):
            self.logger.warning(f"Requested stage {stage_idx} out of range. Using last stage.")
            stage_idx = len(self.stages) - 1
        
        stage_config = self.stages[stage_idx]
        self.logger.info(f"Setting up stage {stage_idx}: {stage_config['name']}")
        
        # Update training parameters for this stage
        self.current_stage = stage_idx
        
        # Create dataloaders for this stage
        self.dataloader = create_dataloader(
            data_root=self.config['data']['processed_data_dir'],
            split='train',
            stage=stage_config['name'],
            batch_size=stage_config['batch_size'],
            config_path=self.config_path
        )
        
        self.val_dataloader = create_dataloader(
            data_root=self.config['data']['processed_data_dir'],
            split='val',
            stage=stage_config['name'],
            batch_size=stage_config['batch_size'],
            shuffle=False,
            config_path=self.config_path
        )
        
        # Stage-specific model adjustments
        if stage_idx == 0:
            # Stage 1: Train with 3D body skeletons and head spheres only
            self._freeze_facial_components()
        elif stage_idx == 1:
            # Stage 2: Add facial representations, train face encoder only
            self._freeze_non_facial_components()
        else:
            # Stage 3: Unfreeze all parameters
            self._unfreeze_all_components()
    
    def _freeze_facial_components(self):
        """Freeze facial guidance components in the model."""
        if hasattr(self.guidance_system, 'facial_encoder'):
            for param in self.guidance_system.facial_encoder.parameters():
                param.requires_grad = False
    
    def _freeze_non_facial_components(self):
        """Freeze non-facial components, leaving facial guidance trainable."""
        # Freeze DiT model except cross-attention layers for facial guidance
        for name, param in self.model.named_parameters():
            if 'cross_attn' not in name:
                param.requires_grad = False
        
        # Freeze non-facial guidance components
        if hasattr(self.guidance_system, 'head_sphere_guidance'):
            for param in self.guidance_system.head_sphere_guidance.parameters():
                param.requires_grad = False
        
        if hasattr(self.guidance_system, 'body_skeleton_guidance'):
            for param in self.guidance_system.body_skeleton_guidance.parameters():
                param.requires_grad = False
        
        # Unfreeze facial encoder
        if hasattr(self.guidance_system, 'facial_encoder'):
            for param in self.guidance_system.facial_encoder.parameters():
                param.requires_grad = True
    
    def _unfreeze_all_components(self):
        """Unfreeze all parameters in the model for full training."""
        # Unfreeze DiT model
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Unfreeze guidance system
        for param in self.guidance_system.parameters():
            param.requires_grad = True
    
    def _noise_images(self, images: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to images according to diffusion timesteps."""
        noise = torch.randn_like(images)
        
        # Get beta schedule
        beta_start = self.config['diffusion']['beta_start']
        beta_end = self.config['diffusion']['beta_end']
        noise_steps = self.config['diffusion']['noise_steps']
        beta_schedule = self.config['diffusion']['beta_schedule']
        
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, noise_steps, device=self.device)
        elif beta_schedule == 'cosine':
            steps = torch.arange(noise_steps + 1, device=self.device) / noise_steps
            alpha_cumprod = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
            alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
            betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
            betas = torch.clamp(betas, max=0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Extract alpha values for the given timesteps
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]])
        
        # Extract the appropriate alpha values for the batch
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        
        # Use gather to get the correct timestep values for the batch
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Expand dimensions for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None]
        
        # Apply noise
        noisy_images = sqrt_alphas_cumprod_t * images + sqrt_one_minus_alphas_cumprod_t * noise
        
        return noisy_images, noise
    
    def _compute_loss(
        self, 
        predicted_noise: torch.Tensor, 
        target_noise: torch.Tensor,
        reference_image: torch.Tensor,
        target_frame: torch.Tensor,
        noisy_image: torch.Tensor,
        timesteps: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components for training."""
        loss_dict = {}
        
        # Basic reconstruction loss
        loss_dict['recon'] = self.losses['reconstruction'](predicted_noise, target_noise)
        
        # Compute perceptual loss if needed
        if 'perceptual_weight' in self.config['loss'] and self.config['loss']['perceptual_weight'] > 0:
            loss_dict['perceptual'] = self.losses['perceptual'](predicted_noise, target_noise)
        
        # Compute identity loss if needed
        if 'identity_weight' in self.config['loss'] and self.config['loss']['identity_weight'] > 0:
            loss_dict['identity'] = self.losses['identity'](reference_image, target_frame)
        
        # Compute temporal consistency loss if needed
        if 'temporal_weight' in self.config['loss'] and self.config['loss']['temporal_weight'] > 0:
            # This would need previous frames in a sequence, which we don't have in this example
            # In a real implementation, we would track previous outputs
            loss_dict['temporal'] = torch.tensor(0.0, device=self.device)
        
        # Compute total loss with weights
        total_loss = self.config['loss']['recon_weight'] * loss_dict['recon']
        
        if 'perceptual' in loss_dict and 'perceptual_weight' in self.config['loss']:
            total_loss += self.config['loss']['perceptual_weight'] * loss_dict['perceptual']
        
        if 'identity' in loss_dict and 'identity_weight' in self.config['loss']:
            total_loss += self.config['loss']['identity_weight'] * loss_dict['identity']
        
        if 'temporal' in loss_dict and 'temporal_weight' in self.config['loss']:
            total_loss += self.config['loss']['temporal_weight'] * loss_dict['temporal']
        
        loss_dict['total'] = total_loss
        
        return loss_dict
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.guidance_system.train()
        
        epoch_losses = {
            'total': 0.0,
            'recon': 0.0,
            'perceptual': 0.0,
            'identity': 0.0,
            'temporal': 0.0
        }
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            reference_image = batch['reference_image'].to(self.device)
            target_frame = batch['target_frame'].to(self.device)
            facial_data = batch['facial_data'].to(self.device) if 'facial_data' in batch else None
            head_sphere = batch['head_sphere'].to(self.device) if 'head_sphere' in batch else None
            body_skeleton = batch['body_skeleton'].to(self.device) if 'body_skeleton' in batch else None
            
            # Generate random timesteps
            batch_size = reference_image.shape[0]
            timesteps = torch.randint(
                0, self.config['diffusion']['noise_steps'], 
                (batch_size,), device=self.device
            ).long()
            
            # Add noise to target frame
            noisy_image, noise = self._noise_images(target_frame, timesteps)
            
            # Generate guidance signals
            guidance_outputs = self.guidance_system(
                facial_images=facial_data.unsqueeze(1) if facial_data is not None else None,
                head_spheres=head_sphere.unsqueeze(1) if head_sphere is not None else None,
                body_skeletons=body_skeleton.unsqueeze(1) if body_skeleton is not None else None,
                reference_frames=reference_image.unsqueeze(1)
            )
            
            # Forward pass with mixed precision if enabled
            if self.fp16:
                with torch.cuda.amp.autocast():
                    # Forward pass through the model
                    predicted_noise = self.model(
                        noisy_image,
                        timesteps,
                        facial_guidance=guidance_outputs.get('facial_tokens', None),
                        head_sphere_guidance=guidance_outputs.get('head_sphere_tokens', None),
                        body_skeleton_guidance=guidance_outputs.get('body_tokens', None)
                    )
                    
                    # Compute losses
                    losses = self._compute_loss(
                        predicted_noise, noise, reference_image, target_frame, noisy_image, timesteps
                    )
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(losses['total']).backward()
                
                # Clip gradients if needed
                if self.gradient_clipping > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.guidance_system.parameters()), 
                        self.gradient_clipping
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                predicted_noise = self.model(
                    noisy_image,
                    timesteps,
                    facial_guidance=guidance_outputs.get('facial_tokens', None),
                    head_sphere_guidance=guidance_outputs.get('head_sphere_tokens', None),
                    body_skeleton_guidance=guidance_outputs.get('body_tokens', None)
                )
                
                # Compute losses
                losses = self._compute_loss(
                    predicted_noise, noise, reference_image, target_frame, noisy_image, timesteps
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                losses['total'].backward()
                
                # Clip gradients if needed
                if self.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.guidance_system.parameters()), 
                        self.gradient_clipping
                    )
                
                self.optimizer.step()
            
            # Update metrics
            for k, v in losses.items():
                if k in epoch_losses:
                    epoch_losses[k] += v.item()
            
            # Log batch metrics
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log metrics and generate samples every N steps
            if self.global_step % self.eval_every == 0:
                log_metrics(
                    {k: v / (batch_idx + 1) for k, v in epoch_losses.items()},
                    self.global_step,
                    prefix='train'
                )
                
                # Generate and log sample images
                with torch.no_grad():
                    self.model.eval()
                    # Here we would have code to generate sample animations from the model
                    # In a real implementation, this would use the inference code
                    self.model.train()
            
            # Save checkpoint every N steps
            if self.global_step % self.save_every == 0:
                checkpoint_path = os.path.join(
                    self.config['training'].get('checkpoint_dir', 'checkpoints'),
                    f"model_step_{self.global_step}.pt"
                )
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                self._save_checkpoint(checkpoint_path)
                self.logger.info(f"Saved checkpoint at step {self.global_step} to {checkpoint_path}")
            
            self.global_step += 1
        
        # Compute average losses for the epoch
        for k in epoch_losses:
            epoch_losses[k] /= len(self.dataloader)
        
        return epoch_losses
    
    def validate(self):
        """Validate the model on the validation set."""
        self.model.eval()
        self.guidance_system.eval()
        
        val_losses = {
            'total': 0.0,
            'recon': 0.0,
            'perceptual': 0.0,
            'identity': 0.0,
            'temporal': 0.0
        }
        
        with torch.no_grad():
            pbar = tqdm(self.val_dataloader, desc="Validation")
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                reference_image = batch['reference_image'].to(self.device)
                target_frame = batch['target_frame'].to(self.device)
                facial_data = batch['facial_data'].to(self.device) if 'facial_data' in batch else None
                head_sphere = batch['head_sphere'].to(self.device) if 'head_sphere' in batch else None
                body_skeleton = batch['body_skeleton'].to(self.device) if 'body_skeleton' in batch else None
                
                # Generate random timesteps
                batch_size = reference_image.shape[0]
                timesteps = torch.randint(
                    0, self.config['diffusion']['noise_steps'], 
                    (batch_size,), device=self.device
                ).long()
                
                # Add noise to target frame
                noisy_image, noise = self._noise_images(target_frame, timesteps)
                
                # Generate guidance signals
                guidance_outputs = self.guidance_system(
                    facial_images=facial_data.unsqueeze(1) if facial_data is not None else None,
                    head_spheres=head_sphere.unsqueeze(1) if head_sphere is not None else None,
                    body_skeletons=body_skeleton.unsqueeze(1) if body_skeleton is not None else None,
                    reference_frames=reference_image.unsqueeze(1)
                )
                
                # Forward pass
                predicted_noise = self.model(
                    noisy_image,
                    timesteps,
                    facial_guidance=guidance_outputs.get('facial_tokens', None),
                    head_sphere_guidance=guidance_outputs.get('head_sphere_tokens', None),
                    body_skeleton_guidance=guidance_outputs.get('body_tokens', None)
                )
                
                # Compute losses
                losses = self._compute_loss(
                    predicted_noise, noise, reference_image, target_frame, noisy_image, timesteps
                )
                
                # Update metrics
                for k, v in losses.items():
                    if k in val_losses:
                        val_losses[k] += v.item()
                
                pbar.set_postfix({'val_loss': losses['total'].item()})
        
        # Compute average losses
        for k in val_losses:
            val_losses[k] /= len(self.val_dataloader)
        
        # Log validation metrics
        log_metrics(val_losses, self.global_step, prefix='val')
        
        # Check if this is the best model
        if val_losses['total'] < self.best_val_loss:
            self.best_val_loss = val_losses['total']
            checkpoint_path = os.path.join(
                self.config['training'].get('checkpoint_dir', 'checkpoints'),
                f"model_step_{self.global_step}.pt"
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            self._save_checkpoint(checkpoint_path, is_best=True)
            self.logger.info(f"New best model with validation loss {self.best_val_loss:.6f}")
        
        return val_losses
    
    def train(self):
        """Main training loop with progressive stages."""
        self.logger.info(f"Starting training with {len(self.stages)} progressive stages")
        
        for stage_idx in range(len(self.stages)):
            # Skip stages we've already completed
            if stage_idx < self.current_stage:
                self.logger.info(f"Skipping completed stage {stage_idx}")
                continue
            
            # Setup the current stage
            self._setup_stage(stage_idx)
            stage_config = self.stages[stage_idx]
            
            self.logger.info(f"Training stage {stage_idx}: {stage_config['name']} for {stage_config['epochs']} epochs")
            
            # Train for the specified number of epochs in this stage
            for epoch in range(self.current_epoch, stage_config['epochs']):
                self.current_epoch = epoch
                self.logger.info(f"Starting epoch {epoch+1}/{stage_config['epochs']} in stage {stage_idx}")
                
                # Train for one epoch
                train_losses = self.train_epoch()
                
                # Validate
                val_losses = self.validate()
                
                # Log epoch summary
                self.logger.info(
                    f"Epoch {epoch+1}/{stage_config['epochs']}, "
                    f"Train Loss: {train_losses['total']:.6f}, "
                    f"Val Loss: {val_losses['total']:.6f}"
                )
                
                # Update learning rate
                if self.lr_scheduler:
                    self.lr_scheduler.step()
            
            # Reset epoch counter for the next stage
            self.current_epoch = 0
            
            # Save stage completion checkpoint
            stage_checkpoint_path = os.path.join(
                self.config['training'].get('checkpoint_dir', 'checkpoints'),
                f"model_stage_{stage_idx}_complete.pt"
            )
            os.makedirs(os.path.dirname(stage_checkpoint_path), exist_ok=True)
            self._save_checkpoint(stage_checkpoint_path)
            self.logger.info(f"Completed stage {stage_idx}, saved checkpoint to {stage_checkpoint_path}")
        
        self.logger.info("Training completed for all stages")


def main(config_path: str, resume_from: Optional[str] = None):
    """Main function to start training."""
    trainer = ChimeraAITrainer(config_path, resume_from)
    trainer.train()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ChimeraAI model")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    main(args.config, args.resume)
