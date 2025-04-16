"""
Diffusion Transformer (DiT) implementation for ChimeraAI.

This module implements the DiT architecture that serves as the foundation
for the human image animation system, replacing the traditional U-Net in
diffusion models with a transformer-based approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, List, Tuple, Optional, Any, Union


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for DiT."""
    
    def __init__(self, dim: int):
        """
        Initialize sinusoidal position embeddings.
        
        Args:
            dim: Dimension of the embeddings
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            time: Time tensor of shape [batch_size]
            
        Returns:
            Position embeddings of shape [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PatchEmbed(nn.Module):
    """Image to Patch Embedding for DiT."""
    
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1024,
    ):
        """
        Initialize patch embedding.
        
        Args:
            img_size: Size of input image
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Dimension of the embeddings
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, img_size, img_size]
            
        Returns:
            Patch embeddings of shape [batch_size, grid_size*grid_size, embed_dim]
        """
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"Input image size ({H}*{W}) doesn't match expected ({self.img_size}*{self.img_size})"
        
        # Project patches
        x = self.proj(x)
        # Flatten spatial dimensions
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class DiTBlock(nn.Module):
    """Diffusion Transformer Block."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        enable_cross_attention: bool = False,
        cross_attention_dim: Optional[int] = None,
    ):
        """
        Initialize DiT block.
        
        Args:
            dim: Dimension of input and output
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            dropout: Dropout probability
            attention_dropout: Attention dropout probability
            enable_cross_attention: Whether to enable cross-attention for guidance
            cross_attention_dim: Dimension of cross-attention input
        """
        super().__init__()
        
        # First normalization and self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # Second normalization and MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Cross-attention for guidance
        self.enable_cross_attention = enable_cross_attention
        if enable_cross_attention:
            assert cross_attention_dim is not None, "cross_attention_dim must be provided when enable_cross_attention is True"
            self.norm_cross = nn.LayerNorm(dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=attention_dropout,
                batch_first=True
            )

    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            context: Optional context tensor for cross-attention of shape [batch_size, context_len, cross_attention_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x
        
        # Cross-attention (if enabled)
        if self.enable_cross_attention and context is not None:
            residual = x
            x = self.norm_cross(x)
            x, _ = self.cross_attn(x, context, context)
            x = residual + x
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) for image generation.
    
    This is the core architecture for ChimeraAI, which replaces the U-Net
    architecture traditionally used in diffusion models with a transformer-based
    approach for better capturing long-range dependencies.
    """
    
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        enable_facial_guidance: bool = True,
        enable_head_sphere_guidance: bool = True,
        enable_body_skeleton_guidance: bool = True,
        facial_embedding_dim: int = 512,
        head_sphere_embedding_dim: int = 512,
        body_skeleton_embedding_dim: int = 512,
    ):
        """
        Initialize DiT.
        
        Args:
            img_size: Size of input image
            patch_size: Size of each patch
            in_channels: Number of input channels
            hidden_size: Dimension of transformer hidden states
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            dropout: Dropout probability
            attention_dropout: Attention dropout probability
            enable_facial_guidance: Whether to enable facial guidance
            enable_head_sphere_guidance: Whether to enable head sphere guidance
            enable_body_skeleton_guidance: Whether to enable body skeleton guidance
            facial_embedding_dim: Dimension of facial embeddings
            head_sphere_embedding_dim: Dimension of head sphere embeddings
            body_skeleton_embedding_dim: Dimension of body skeleton embeddings
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size
        )
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Time embeddings
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Guidance embeddings and projections
        self.enable_facial_guidance = enable_facial_guidance
        self.enable_head_sphere_guidance = enable_head_sphere_guidance
        self.enable_body_skeleton_guidance = enable_body_skeleton_guidance
        
        if enable_facial_guidance:
            self.facial_projection = nn.Linear(facial_embedding_dim, hidden_size)
        
        if enable_head_sphere_guidance:
            self.head_sphere_projection = nn.Linear(head_sphere_embedding_dim, hidden_size)
        
        if enable_body_skeleton_guidance:
            self.body_skeleton_projection = nn.Linear(body_skeleton_embedding_dim, hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                enable_cross_attention=True,
                cross_attention_dim=hidden_size
            )
            for _ in range(depth)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, patch_size * patch_size * in_channels),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        """Initialize weights for transformer layers."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        facial_guidance: Optional[torch.Tensor] = None,
        head_sphere_guidance: Optional[torch.Tensor] = None,
        body_skeleton_guidance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, img_size, img_size]
            timesteps: Diffusion timesteps of shape [batch_size]
            facial_guidance: Optional facial guidance tensor
            head_sphere_guidance: Optional head sphere guidance tensor
            body_skeleton_guidance: Optional body skeleton guidance tensor
            
        Returns:
            Output tensor of shape [batch_size, in_channels, img_size, img_size]
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Get time embeddings
        time_emb = self.time_embed(timesteps)
        
        # Process guidance inputs
        context_embeddings = []
        context_masks = []
        
        if self.enable_facial_guidance and facial_guidance is not None:
            facial_emb = self.facial_projection(facial_guidance)
            context_embeddings.append(facial_emb)
            context_masks.append(torch.ones(B, facial_emb.shape[1], device=x.device))
        
        if self.enable_head_sphere_guidance and head_sphere_guidance is not None:
            head_sphere_emb = self.head_sphere_projection(head_sphere_guidance)
            context_embeddings.append(head_sphere_emb)
            context_masks.append(torch.ones(B, head_sphere_emb.shape[1], device=x.device))
        
        if self.enable_body_skeleton_guidance and body_skeleton_guidance is not None:
            body_skeleton_emb = self.body_skeleton_projection(body_skeleton_guidance)
            context_embeddings.append(body_skeleton_emb)
            context_masks.append(torch.ones(B, body_skeleton_emb.shape[1], device=x.device))
        
        # Combine all guidance context
        if context_embeddings:
            context = torch.cat(context_embeddings, dim=1)
        else:
            context = None
        
        # Add time embeddings to each token
        time_emb = time_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
        x = x + time_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, context)
        
        # Final layer normalization
        x = self.norm(x)
        
        # Project to patch output
        x = self.out_proj(x)
        
        # Reshape to image
        x = rearrange(
            x,
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=self.grid_size,
            w=self.grid_size,
            p1=self.patch_size,
            p2=self.patch_size
        )
        
        return x


def create_dit_model(config: Dict) -> DiT:
    """
    Create a DiT model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized DiT model
    """
    return DiT(
        img_size=config.get('img_size', 512),
        patch_size=config.get('patch_size', 2),
        in_channels=config.get('in_channels', 3),
        hidden_size=config.get('hidden_size', 1024),
        depth=config.get('depth', 24),
        num_heads=config.get('num_heads', 16),
        mlp_ratio=config.get('mlp_ratio', 4.0),
        dropout=config.get('dropout', 0.1),
        attention_dropout=config.get('attention_dropout', 0.1),
        enable_facial_guidance=config.get('enable_facial_guidance', True),
        enable_head_sphere_guidance=config.get('enable_head_sphere_guidance', True),
        enable_body_skeleton_guidance=config.get('enable_body_skeleton_guidance', True),
    )
