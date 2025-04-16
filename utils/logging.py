"""
Logging utilities for ChimeraAI.

This module provides functions for logging training progress and results.
It implements both file-based logging and integration with Weights & Biases.
"""

import os
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Try to import wandb, but don't fail if it's not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup a logger with console and file handlers.
    
    Args:
        name: Name of the logger
        log_dir: Directory to save log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{name.lower()}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class Logger:
    """
    Logger class for training and evaluation.
    
    This class provides a unified interface for logging to console,
    files, TensorBoard, and Weights & Biases.
    """
    
    def __init__(
        self,
        name: str,
        log_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        resume_wandb: Optional[str] = None
    ):
        """
        Initialize logger.
        
        Args:
            name: Name of the experiment
            log_dir: Directory to save logs
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_project: Weights & Biases project name
            wandb_config: Weights & Biases configuration
            resume_wandb: Weights & Biases run ID to resume
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup text logger
        self.logger = setup_logger(name, log_dir)
        
        # Setup TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        
        # Setup Weights & Biases
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            if WANDB_AVAILABLE:
                if resume_wandb:
                    wandb.init(
                        project=wandb_project,
                        config=wandb_config,
                        name=name,
                        dir=str(self.log_dir),
                        resume=resume_wandb
                    )
                else:
                    wandb.init(
                        project=wandb_project,
                        config=wandb_config,
                        name=name,
                        dir=str(self.log_dir)
                    )
            else:
                self.logger.warning("Weights & Biases (wandb) not installed, falling back to local logging only")
                self.use_wandb = False
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: Optional[str] = None
    ) -> None:
        """
        Log scalar metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step
            prefix: Optional prefix for metric names
        """
        # Add prefix if provided
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metrics_str}")
        
        # Log to TensorBoard
        if self.use_tensorboard:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)
        
        # Log to Weights & Biases
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_images(
        self,
        images: Dict[str, Union[torch.Tensor, np.ndarray]],
        step: int,
        prefix: Optional[str] = None
    ) -> None:
        """
        Log images.
        
        Args:
            images: Dictionary of image names and tensors
            step: Training step
            prefix: Optional prefix for image names
        """
        # Add prefix if provided
        if prefix:
            images = {f"{prefix}/{k}": v for k, v in images.items()}
        
        # Convert tensors to numpy arrays
        processed_images = {}
        for k, v in images.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 4:  # [N, C, H, W]
                    v = v.detach().cpu().numpy()
                elif v.ndim == 3 and v.shape[0] in [1, 3]:  # [C, H, W]
                    v = v.unsqueeze(0).detach().cpu().numpy()
                else:
                    raise ValueError(f"Unsupported image tensor shape: {v.shape}")
            
            # Ensure values are in [0, 1]
            if v.max() > 1.0:
                v = v / 255.0
            
            processed_images[k] = v
        
        # Log to TensorBoard
        if self.use_tensorboard:
            for k, v in processed_images.items():
                # TensorBoard expects [N, C, H, W] with values in [0, 1]
                self.tb_writer.add_images(k, v, step)
        
        # Log to Weights & Biases
        if self.use_wandb:
            # wandb expects [N, H, W, C] with values in [0, 255]
            wandb_images = {}
            for k, v in processed_images.items():
                v = np.transpose(v, (0, 2, 3, 1))  # [N, C, H, W] -> [N, H, W, C]
                v = (v * 255).astype(np.uint8)
                wandb_images[k] = [wandb.Image(img) for img in v]
            
            wandb.log(wandb_images, step=step)
    
    def log_video(
        self,
        videos: Dict[str, Union[torch.Tensor, np.ndarray]],
        step: int,
        fps: int = 30,
        prefix: Optional[str] = None
    ) -> None:
        """
        Log videos.
        
        Args:
            videos: Dictionary of video names and tensors
            step: Training step
            fps: Frames per second
            prefix: Optional prefix for video names
        """
        # Add prefix if provided
        if prefix:
            videos = {f"{prefix}/{k}": v for k, v in videos.items()}
        
        # Convert tensors to numpy arrays
        processed_videos = {}
        for k, v in videos.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 5:  # [N, T, C, H, W]
                    v = v.detach().cpu().numpy()
                elif v.ndim == 4 and v.shape[1] in [1, 3]:  # [T, C, H, W]
                    v = v.unsqueeze(0).detach().cpu().numpy()
                else:
                    raise ValueError(f"Unsupported video tensor shape: {v.shape}")
            
            # Ensure values are in [0, 1]
            if v.max() > 1.0:
                v = v / 255.0
            
            processed_videos[k] = v
        
        # Log to TensorBoard
        if self.use_tensorboard:
            for k, v in processed_videos.items():
                # TensorBoard expects [N, T, C, H, W] with values in [0, 1]
                # self.tb_writer.add_video(k, v, step, fps=fps)
                # Since TensorBoard's add_video might not be compatible with all setups,
                # we'll log the first and last frames as images instead
                if v.shape[1] > 0:
                    self.tb_writer.add_images(f"{k}_first", v[:, 0], step)
                if v.shape[1] > 1:
                    self.tb_writer.add_images(f"{k}_last", v[:, -1], step)
        
        # Log to Weights & Biases
        if self.use_wandb:
            # wandb expects [N, T, H, W, C] with values in [0, 255]
            wandb_videos = {}
            for k, v in processed_videos.items():
                v = np.transpose(v, (0, 1, 3, 4, 2))  # [N, T, C, H, W] -> [N, T, H, W, C]
                v = (v * 255).astype(np.uint8)
                
                # Only log the first video in the batch to save bandwidth
                wandb_videos[k] = wandb.Video(v[0], fps=fps, format="mp4")
            
            wandb.log(wandb_videos, step=step)
    
    def log_model_graph(self, model: torch.nn.Module, input_shape: List[int]) -> None:
        """
        Log model architecture graph.
        
        Args:
            model: PyTorch model
            input_shape: Shape of input tensor [N, C, H, W]
        """
        if self.use_tensorboard:
            # Create dummy input
            device = next(model.parameters()).device
            dummy_input = torch.zeros(input_shape, device=device)
            
            try:
                self.tb_writer.add_graph(model, dummy_input)
            except Exception as e:
                self.logger.warning(f"Failed to log model graph: {e}")
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
        """
        # Log to console
        self.logger.info(f"Hyperparameters: {json.dumps(params, indent=2)}")
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.tb_writer.add_text("hyperparameters", json.dumps(params, indent=2))
        
        # Log to Weights & Biases
        if self.use_wandb and not wandb.run.config:
            wandb.config.update(params)
    
    def close(self) -> None:
        """Close all loggers."""
        if self.use_tensorboard:
            self.tb_writer.close()
        
        if self.use_wandb:
            wandb.finish()


# Global logger instance for convenient access
_logger = None


def get_logger() -> Logger:
    """
    Get the global logger instance.
    
    Returns:
        Global logger instance
    """
    global _logger
    if _logger is None:
        raise RuntimeError("Logger not initialized. Call init_logger first.")
    return _logger


def init_logger(
    name: str,
    log_dir: str,
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_config: Optional[Dict] = None,
    resume_wandb: Optional[str] = None
) -> Logger:
    """
    Initialize the global logger.
    
    Args:
        name: Name of the experiment
        log_dir: Directory to save logs
        use_tensorboard: Whether to use TensorBoard
        use_wandb: Whether to use Weights & Biases
        wandb_project: Weights & Biases project name
        wandb_config: Weights & Biases configuration
        resume_wandb: Weights & Biases run ID to resume
        
    Returns:
        Global logger instance
    """
    global _logger
    _logger = Logger(
        name=name,
        log_dir=log_dir,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_config=wandb_config,
        resume_wandb=resume_wandb
    )
    return _logger


# Convenience functions that use the global logger

def log_metrics(
    metrics: Dict[str, float],
    step: int,
    prefix: Optional[str] = None
) -> None:
    """Log metrics using the global logger."""
    get_logger().log_metrics(metrics, step, prefix)


def log_images(
    images: Dict[str, Union[torch.Tensor, np.ndarray]],
    step: int,
    prefix: Optional[str] = None
) -> None:
    """Log images using the global logger."""
    get_logger().log_images(images, step, prefix)


def log_video(
    videos: Dict[str, Union[torch.Tensor, np.ndarray]],
    step: int,
    fps: int = 30,
    prefix: Optional[str] = None
) -> None:
    """Log videos using the global logger."""
    get_logger().log_video(videos, step, fps, prefix)


def log_model_graph(model: torch.nn.Module, input_shape: List[int]) -> None:
    """Log model graph using the global logger."""
    get_logger().log_model_graph(model, input_shape)


def log_hyperparams(params: Dict[str, Any]) -> None:
    """Log hyperparameters using the global logger."""
    get_logger().log_hyperparams(params)
