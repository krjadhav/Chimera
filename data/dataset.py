"""
Dataset implementation for ChimeraAI.

This module provides dataset classes for loading and preprocessing
input images and motion data with different scales and poses.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import h5py
from typing import Dict, List, Tuple, Optional, Union

from utils.config import load_config


class ChimeraAIDataset(Dataset):
    """
    Dataset for training and evaluating the ChimeraAI model.
    
    This dataset handles different types of input data:
    - Reference images
    - Facial representations
    - 3D head spheres
    - 3D body skeletons
    - Target frames
    """
    
    def __init__(
        self, 
        data_root: str,
        split: str = "train",
        stage: str = "full_body",
        config_path: str = "configs/data_preprocessing.yaml",
        transform = None
    ):
        """
        Initialize the ChimeraAI dataset.
        
        Args:
            data_root: Root directory containing processed data
            split: Dataset split ('train', 'val', or 'test')
            stage: Training stage ('portrait', 'upper_body', or 'full_body')
            config_path: Path to the data configuration file
            transform: Optional transforms to apply to the data
        """
        self.data_root = data_root
        self.split = split
        self.stage = stage
        self.transform = transform
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Set image size based on stage
        if stage == "portrait":
            self.image_size = 256
        elif stage == "upper_body":
            self.image_size = 384
        elif stage == "full_body":
            self.image_size = 512
        else:
            raise ValueError(f"Invalid stage: {stage}")
        
        # Load dataset index
        self.samples = self._load_dataset_index()
    
    def _load_dataset_index(self) -> List[Dict]:
        """
        Load the dataset index, which contains paths and metadata for each sample.
        
        Returns:
            List of dictionaries, where each dictionary contains paths and metadata for a sample
        """
        index_file = os.path.join(self.data_root, f"{self.split}_{self.stage}_index.h5")
        
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        samples = []
        with h5py.File(index_file, 'r') as f:
            num_samples = f['num_samples'][()]
            
            for i in range(num_samples):
                sample = {
                    'reference_image': f[f'sample_{i}/reference_image'][()].decode('utf-8'),
                    'target_frames': [f[f'sample_{i}/target_frames/{j}'][()].decode('utf-8') 
                                     for j in range(f[f'sample_{i}/num_target_frames'][()])],
                    'facial_data': f[f'sample_{i}/facial_data'][()].decode('utf-8') if 'facial_data' in f[f'sample_{i}'] else None,
                    'head_sphere': f[f'sample_{i}/head_sphere'][()].decode('utf-8') if 'head_sphere' in f[f'sample_{i}'] else None,
                    'body_skeleton': f[f'sample_{i}/body_skeleton'][()].decode('utf-8') if 'body_skeleton' in f[f'sample_{i}'] else None,
                    'metadata': {k: f[f'sample_{i}/metadata/{k}'][()] for k in f[f'sample_{i}/metadata'].keys()}
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def _load_image(self, path: str) -> torch.Tensor:
        """
        Load and preprocess an image.
        
        Args:
            path: Path to the image file
            
        Returns:
            Preprocessed image tensor with shape [C, H, W]
        """
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            img = np.array(img) / 255.0  # Normalize to [0, 1]
            img = torch.from_numpy(img).permute(2, 0, 1).float()  # [C, H, W]
            
            if self.transform:
                img = self.transform(img)
                
            return img
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a blank image if loading fails
            return torch.zeros(3, self.image_size, self.image_size)
    
    def _load_facial_data(self, path: str) -> torch.Tensor:
        """
        Load facial representation data.
        
        Args:
            path: Path to the facial data file
            
        Returns:
            Facial representation tensor
        """
        if path is None:
            # Return zero tensor if no data is available
            return torch.zeros(68, 2)
        
        try:
            with np.load(path) as data:
                facial_data = data['landmarks']
            return torch.from_numpy(facial_data).float()
        except Exception as e:
            print(f"Error loading facial data {path}: {e}")
            return torch.zeros(68, 2)
    
    def _load_head_sphere(self, path: str) -> torch.Tensor:
        """
        Load 3D head sphere data.
        
        Args:
            path: Path to the head sphere data file
            
        Returns:
            Head sphere representation tensor
        """
        if path is None:
            # Return zero tensor if no data is available
            resolution = self.config['preprocessing']['head_sphere']['resolution']
            return torch.zeros(resolution, resolution, resolution)
        
        try:
            with np.load(path) as data:
                head_sphere = data['sphere']
            return torch.from_numpy(head_sphere).float()
        except Exception as e:
            print(f"Error loading head sphere {path}: {e}")
            resolution = self.config['preprocessing']['head_sphere']['resolution']
            return torch.zeros(resolution, resolution, resolution)
    
    def _load_body_skeleton(self, path: str) -> torch.Tensor:
        """
        Load 3D body skeleton data.
        
        Args:
            path: Path to the body skeleton data file
            
        Returns:
            Body skeleton representation tensor
        """
        if path is None:
            # Return zero tensor if no data is available
            num_keypoints = self.config['preprocessing']['body_skeleton']['keypoints']
            return torch.zeros(num_keypoints, 3)
        
        try:
            with np.load(path) as data:
                body_skeleton = data['keypoints']
            return torch.from_numpy(body_skeleton).float()
        except Exception as e:
            print(f"Error loading body skeleton {path}: {e}")
            num_keypoints = self.config['preprocessing']['body_skeleton']['keypoints']
            return torch.zeros(num_keypoints, 3)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing all data for the sample
        """
        sample = self.samples[idx]
        
        # Load reference image
        reference_image = self._load_image(sample['reference_image'])
        
        # Load target frames (select one randomly for training)
        target_idx = np.random.randint(0, len(sample['target_frames']))
        target_frame = self._load_image(sample['target_frames'][target_idx])
        
        # Load motion guidance data
        facial_data = self._load_facial_data(sample['facial_data'])
        head_sphere = self._load_head_sphere(sample['head_sphere'])
        body_skeleton = self._load_body_skeleton(sample['body_skeleton'])
        
        # Create sample dictionary
        return {
            'reference_image': reference_image,
            'target_frame': target_frame,
            'facial_data': facial_data,
            'head_sphere': head_sphere,
            'body_skeleton': body_skeleton,
            'metadata': sample['metadata']
        }


def create_dataloader(
    data_root: str,
    split: str,
    stage: str,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    config_path: str = "configs/data_preprocessing.yaml"
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader for the ChimeraAI dataset.
    
    Args:
        data_root: Root directory containing processed data
        split: Dataset split ('train', 'val', or 'test')
        stage: Training stage ('portrait', 'upper_body', or 'full_body')
        batch_size: Batch size for the dataloader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the dataset
        config_path: Path to the data configuration file
        
    Returns:
        DataLoader for the specified dataset
    """
    dataset = ChimeraAIDataset(
        data_root=data_root,
        split=split,
        stage=stage,
        config_path=config_path
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
