"""
Loss functions for ChimeraAI training.

This module implements various loss functions used in training the ChimeraAI model,
including reconstruction loss, perceptual loss, identity loss, and temporal consistency loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Dict, Tuple, Optional, Union


class ReconstructionLoss(nn.Module):
    """
    Basic reconstruction loss for diffusion model training.
    
    This loss compares the predicted noise with the actual noise added during
    the diffusion process, using L2 (MSE) loss.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize reconstruction loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            predicted: Predicted noise tensor
            target: Target noise tensor
            
        Returns:
            Loss value
        """
        return self.loss_fn(predicted, target)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG16 features.
    
    This loss compares the high-level feature representations of images,
    rather than pixel-level differences, to better preserve perceptual quality.
    """
    
    def __init__(
        self, 
        layers: List[int] = [3, 8, 15, 22],
        weights: List[float] = [1.0, 1.0, 1.0, 1.0],
        normalize: bool = True
    ):
        """
        Initialize perceptual loss.
        
        Args:
            layers: VGG16 layer indices to extract features from
            weights: Weights for each layer's contribution
            normalize: Whether to normalize features
        """
        super().__init__()
        
        # Load pretrained VGG16 model
        vgg = models.vgg16(pretrained=True).features
        vgg.eval()
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.model = vgg
        self.layers = layers
        self.weights = weights
        self.normalize = normalize
        
        # Register mean and std for normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor for VGG."""
        return (x - self.mean) / self.std
    
    def _get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from specified VGG layers."""
        features = []
        
        # Normalize if required
        if self.normalize:
            x = self._normalize(x)
        
        # Extract features from each specified layer
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        
        return features
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            predicted: Predicted image tensor
            target: Target image tensor
            
        Returns:
            Loss value
        """
        # Get features
        predicted_features = self._get_features(predicted)
        target_features = self._get_features(target)
        
        # Compute loss
        loss = 0.0
        for i, (p, t) in enumerate(zip(predicted_features, target_features)):
            loss += self.weights[i] * F.mse_loss(p, t)
        
        return loss


class IdentityLoss(nn.Module):
    """
    Identity preservation loss.
    
    This loss ensures that the animated images maintain the identity features
    of the reference image, which is crucial for realistic human animation.
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize identity loss.
        
        Args:
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Use a face recognition model to extract identity features
        # In a real implementation, this would be a proper face recognition model
        # like ArcFace, FaceNet, or similar
        # For simplicity, we'll use ResNet18 here as a placeholder
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layer
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Register mean and std for normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor for ResNet."""
        return (x - self.mean) / self.std
    
    def _extract_identity(self, x: torch.Tensor) -> torch.Tensor:
        """Extract identity features from an image."""
        # Normalize
        x = self._normalize(x)
        
        # Extract features
        features = self.model(x)
        
        # Flatten
        features = features.view(features.size(0), -1)
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def forward(self, reference: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        """
        Compute identity loss.
        
        Args:
            reference: Reference image tensor
            generated: Generated image tensor
            
        Returns:
            Loss value
        """
        # Extract identity features
        reference_features = self._extract_identity(reference)
        generated_features = self._extract_identity(generated)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(reference_features, generated_features)
        
        # Convert to loss (1 - similarity)
        loss = 1.0 - similarity.mean()
        
        return loss


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for smooth animations.
    
    This loss enforces smooth transitions between consecutive frames in an animation,
    penalizing sudden changes in appearance and motion.
    """
    
    def __init__(
        self,
        use_flow: bool = True,
        flow_weight: float = 1.0,
        rgb_weight: float = 0.5
    ):
        """
        Initialize temporal consistency loss.
        
        Args:
            use_flow: Whether to use optical flow for consistency
            flow_weight: Weight for flow-based consistency
            rgb_weight: Weight for direct RGB consistency
        """
        super().__init__()
        
        self.use_flow = use_flow
        self.flow_weight = flow_weight
        self.rgb_weight = rgb_weight
        
        # In a real implementation, we would initialize an optical flow model here
        # For simplicity, we'll omit the actual flow computation
    
    def _compute_flow(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute optical flow between two frames.
        
        This is a placeholder for actual optical flow computation.
        In a real implementation, this would use a proper optical flow model.
        """
        # Placeholder implementation
        # Returns a random flow field for demonstration
        b, c, h, w = x1.shape
        return torch.randn(b, 2, h, w, device=x1.device)
    
    def _warp(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp an image according to optical flow.
        
        Args:
            x: Image tensor to warp
            flow: Optical flow field
            
        Returns:
            Warped image
        """
        # Create sampling grid
        b, c, h, w = x.shape
        
        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=x.device),
            torch.arange(0, w, device=x.device)
        )
        grid = torch.stack((grid_x, grid_y), dim=0).float()
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
        
        # Add flow to grid
        grid = grid + flow
        
        # Normalize grid to [-1, 1] for grid_sample
        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / (w - 1) - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / (h - 1) - 1.0
        
        # Permute grid for grid_sample
        grid = grid.permute(0, 2, 3, 1)
        
        # Warp image
        warped = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped
    
    def forward(
        self,
        current_frame: torch.Tensor,
        previous_frame: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            current_frame: Current frame tensor
            previous_frame: Previous frame tensor
            mask: Optional mask to focus on specific regions
            
        Returns:
            Loss value
        """
        # Direct RGB consistency
        rgb_diff = torch.abs(current_frame - previous_frame)
        rgb_loss = rgb_diff.mean()
        
        # Flow-based consistency
        if self.use_flow:
            # Compute flow from previous to current
            flow = self._compute_flow(previous_frame, current_frame)
            
            # Warp previous frame to current
            warped_previous = self._warp(previous_frame, flow)
            
            # Compute difference
            flow_diff = torch.abs(current_frame - warped_previous)
            
            # Apply mask if provided
            if mask is not None:
                flow_diff = flow_diff * mask
            
            flow_loss = flow_diff.mean()
        else:
            flow_loss = torch.tensor(0.0, device=current_frame.device)
        
        # Combine losses
        total_loss = self.rgb_weight * rgb_loss + self.flow_weight * flow_loss
        
        return total_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for ChimeraAI training.
    
    This loss combines multiple loss terms with configurable weights.
    """
    
    def __init__(
        self,
        recon_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        identity_weight: float = 0.5,
        temporal_weight: float = 0.3
    ):
        """
        Initialize combined loss.
        
        Args:
            recon_weight: Weight for reconstruction loss
            perceptual_weight: Weight for perceptual loss
            identity_weight: Weight for identity loss
            temporal_weight: Weight for temporal consistency loss
        """
        super().__init__()
        
        self.recon_loss = ReconstructionLoss()
        self.perceptual_loss = PerceptualLoss()
        self.identity_loss = IdentityLoss()
        self.temporal_loss = TemporalConsistencyLoss()
        
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.identity_weight = identity_weight
        self.temporal_weight = temporal_weight
    
    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        reference_image: torch.Tensor,
        current_frame: torch.Tensor,
        previous_frame: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            predicted_noise: Predicted noise tensor
            target_noise: Target noise tensor
            reference_image: Reference image tensor
            current_frame: Current frame tensor
            previous_frame: Optional previous frame tensor for temporal consistency
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses = {}
        
        # Reconstruction loss
        losses['recon'] = self.recon_loss(predicted_noise, target_noise)
        
        # Perceptual loss
        losses['perceptual'] = self.perceptual_loss(predicted_noise, target_noise)
        
        # Identity loss
        losses['identity'] = self.identity_loss(reference_image, current_frame)
        
        # Temporal consistency loss
        if previous_frame is not None:
            losses['temporal'] = self.temporal_loss(current_frame, previous_frame)
        else:
            losses['temporal'] = torch.tensor(0.0, device=current_frame.device)
        
        # Compute total loss
        total_loss = (
            self.recon_weight * losses['recon'] +
            self.perceptual_weight * losses['perceptual'] +
            self.identity_weight * losses['identity'] +
            self.temporal_weight * losses['temporal']
        )
        
        losses['total'] = total_loss
        
        return total_loss, losses
