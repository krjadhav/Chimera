"""
Visualization utilities for ChimeraAI.

This module provides functions for visualizing model outputs, creating animations,
and generating comparison visualizations.
"""

import os
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from PIL import Image


def tensor_to_image(
    tensor: Union[np.ndarray, 'torch.Tensor'],
    normalize: bool = True,
    denormalize: bool = False
) -> np.ndarray:
    """
    Convert a tensor to a numpy image.
    
    Args:
        tensor: Input tensor in format [C, H, W] or [H, W, C]
        normalize: Whether to normalize to [0, 255] range
        denormalize: Whether to denormalize from [-1, 1] to [0, 1]
        
    Returns:
        Image as numpy array in [H, W, C] format with values in [0, 255]
    """
    # Convert torch tensor to numpy if needed
    if 'torch.Tensor' in str(type(tensor)):
        tensor = tensor.detach().cpu().numpy()
    
    # Handle different tensor formats
    if tensor.shape[0] in [1, 3] and len(tensor.shape) == 3:
        # [C, H, W] format, convert to [H, W, C]
        tensor = np.transpose(tensor, (1, 2, 0))
    
    # Handle single-channel images
    if tensor.shape[-1] == 1:
        tensor = np.repeat(tensor, 3, axis=-1)
    
    # Denormalize from [-1, 1] to [0, 1] if needed
    if denormalize:
        tensor = (tensor + 1) / 2.0
    
    # Normalize to [0, 255] if needed
    if normalize:
        if tensor.max() <= 1.0:
            tensor = tensor * 255.0
        tensor = np.clip(tensor, 0, 255).astype(np.uint8)
    
    return tensor


def create_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30,
    is_gif: bool = False
) -> None:
    """
    Create a video or GIF from a list of frames.
    
    Args:
        frames: List of frames as numpy arrays
        output_path: Path to save the output file
        fps: Frames per second
        is_gif: Whether to create a GIF instead of a video
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if is_gif:
        # Create GIF using PIL
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            optimize=False,
            duration=int(1000 / fps),
            loop=0
        )
    else:
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        # Release resources
        video_writer.release()


def save_image_grid(
    images: List[np.ndarray],
    output_path: str,
    rows: int = 1,
    cols: Optional[int] = None,
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 12)
) -> None:
    """
    Save a grid of images.
    
    Args:
        images: List of images as numpy arrays
        output_path: Path to save the output image
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        titles: Optional list of titles for each image
        figsize: Figure size in inches
    """
    n_images = len(images)
    
    # Calculate number of columns if not specified
    if cols is None:
        cols = (n_images + rows - 1) // rows
    
    # Create figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single row or column case
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot images
    for i, ax in enumerate(axes.flat):
        if i < n_images:
            # Get image
            img = images[i]
            
            # Handle grayscale images
            if len(img.shape) == 2 or img.shape[2] == 1:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            
            # Add title if provided
            if titles and i < len(titles):
                ax.set_title(titles[i])
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def visualize_motion_guidance(
    image: np.ndarray,
    facial_landmarks: Optional[np.ndarray] = None,
    body_skeleton: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    show: bool = False
) -> np.ndarray:
    """
    Visualize motion guidance on an image.
    
    Args:
        image: Input image
        facial_landmarks: Facial landmarks with shape [N, 2]
        body_skeleton: Body skeleton keypoints with shape [N, 2] or [N, 3]
        output_path: Path to save the output image
        show: Whether to display the image
        
    Returns:
        Visualization image
    """
    # Make a copy of the image to avoid modifying the original
    vis_image = image.copy()
    
    # Convert to BGR for OpenCV drawing
    if vis_image.shape[2] == 3:
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    else:
        vis_image_bgr = vis_image
    
    # Draw facial landmarks if provided
    if facial_landmarks is not None:
        for i, (x, y) in enumerate(facial_landmarks):
            # Draw landmark points
            cv2.circle(
                vis_image_bgr,
                (int(x), int(y)),
                1,
                (0, 255, 0),
                -1
            )
            
            # Connect landmarks to form face contour
            # This is a simplified version - in a real implementation, 
            # proper connections based on facial landmark indices would be used
            if i > 0:
                cv2.line(
                    vis_image_bgr,
                    (int(facial_landmarks[i-1][0]), int(facial_landmarks[i-1][1])),
                    (int(x), int(y)),
                    (0, 255, 0),
                    1
                )
    
    # Draw body skeleton if provided
    if body_skeleton is not None:
        # Define connections between keypoints
        # This is a simplified COCO-format skeleton
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head and arms
            (0, 5), (0, 6), (5, 7), (6, 8),  # Torso and legs
            (1, 2), (5, 6)  # Shoulders and hips
        ]
        
        # Extract 2D coordinates and confidences
        if body_skeleton.shape[1] >= 3:
            points = body_skeleton[:, :2]
            confidences = body_skeleton[:, 2]
        else:
            points = body_skeleton
            confidences = np.ones(len(body_skeleton))
        
        # Draw keypoints
        for i, ((x, y), conf) in enumerate(zip(points, confidences)):
            if conf > 0.1:  # Only draw if confidence is high enough
                cv2.circle(
                    vis_image_bgr,
                    (int(x), int(y)),
                    3,
                    (0, 0, 255),
                    -1
                )
        
        # Draw connections
        for (i, j) in connections:
            if i < len(points) and j < len(points):
                if confidences[i] > 0.1 and confidences[j] > 0.1:
                    cv2.line(
                        vis_image_bgr,
                        (int(points[i][0]), int(points[i][1])),
                        (int(points[j][0]), int(points[j][1])),
                        (255, 0, 0),
                        2
                    )
    
    # Convert back to RGB
    if vis_image.shape[2] == 3:
        vis_image = cv2.cvtColor(vis_image_bgr, cv2.COLOR_BGR2RGB)
    else:
        vis_image = vis_image_bgr
    
    # Save image if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    # Show image if requested
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.show()
    
    return vis_image


def visualize_animation_comparison(
    source_frames: List[np.ndarray],
    generated_frames: List[np.ndarray],
    reference_image: np.ndarray,
    output_path: str,
    frame_indices: Optional[List[int]] = None,
    num_frames: int = 6
) -> None:
    """
    Create a comparison visualization of source and generated frames.
    
    Args:
        source_frames: List of source frames
        generated_frames: List of generated frames
        reference_image: Reference image
        output_path: Path to save the output image
        frame_indices: Specific frame indices to visualize
        num_frames: Number of frames to show if indices not provided
    """
    # Validate inputs
    assert len(source_frames) == len(generated_frames), "Number of source and generated frames must match"
    
    # Select frames to visualize
    if frame_indices is None:
        total_frames = len(source_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    else:
        num_frames = len(frame_indices)
    
    # Create visualization
    rows = 2  # Source and generated
    cols = num_frames + 1  # Frames + reference
    figsize = (cols * 4, rows * 4)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Add title
    fig.suptitle("Animation Comparison", fontsize=16)
    
    # Show reference image
    axes[0, 0].imshow(reference_image)
    axes[0, 0].set_title("Reference")
    axes[1, 0].imshow(reference_image)
    axes[1, 0].set_title("Reference")
    
    # Show selected frames
    for i, frame_idx in enumerate(frame_indices):
        col = i + 1
        
        # Source frame
        axes[0, col].imshow(source_frames[frame_idx])
        axes[0, col].set_title(f"Source {frame_idx}")
        
        # Generated frame
        axes[1, col].imshow(generated_frames[frame_idx])
        axes[1, col].set_title(f"Generated {frame_idx}")
    
    # Remove ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add row labels
    axes[0, 0].set_ylabel("Source", fontsize=14)
    axes[1, 0].set_ylabel("Generated", fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def visualize_attention_maps(
    attention_maps: np.ndarray,
    image: np.ndarray,
    output_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Visualize attention maps.
    
    Args:
        attention_maps: Attention maps with shape [N, H, W]
        image: Input image
        output_path: Path to save the output image
        show: Whether to display the image
    """
    num_maps = attention_maps.shape[0]
    
    # Create grid layout based on number of maps
    cols = int(np.ceil(np.sqrt(num_maps + 1)))
    rows = int(np.ceil((num_maps + 1) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    # Handle single row or column case
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Show original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    
    # Show attention maps
    for i in range(num_maps):
        row = (i + 1) // cols
        col = (i + 1) % cols
        
        attention = attention_maps[i]
        
        # Normalize for visualization
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        # Use 'jet' colormap for better visibility
        axes[row, col].imshow(image, alpha=0.6)
        im = axes[row, col].imshow(attention, cmap='jet', alpha=0.4)
        axes[row, col].set_title(f"Attention {i+1}")
    
    # Remove ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide empty subplots
    for i in range(num_maps + 1, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    # Show figure
    if show:
        plt.show()
    else:
        plt.close(fig)
