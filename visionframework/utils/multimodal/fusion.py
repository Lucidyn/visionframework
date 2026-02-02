"""
Multimodal fusion functionality

This module provides functionality for fusing information from different modalities.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FusionType(Enum):
    """Fusion type enum"""
    CONCAT = "concat"
    ADD = "add"
    MULTIPLY = "multiply"
    BILINEAR = "bilinear"
    ATTENTION = "attention"
    CROSS_ATTENTION = "cross_attention"
    GATE = "gate"
    DENSE = "dense"


class ModalityType(Enum):
    """Modality type enum"""
    VISION = "vision"
    LANGUAGE = "language"
    AUDIO = "audio"
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    SENSOR = "sensor"


@dataclass
class FusionConfig:
    """
    Fusion configuration class
    
    Attributes:
        fusion_type: Type of fusion to use
        input_dims: List of input dimensions for each modality
        hidden_dim: Hidden dimension for fusion
        output_dim: Output dimension after fusion
        num_heads: Number of attention heads (for attention-based fusion)
        dropout: Dropout rate
        device: Device to use
    """
    fusion_type: FusionType
    input_dims: List[int]
    hidden_dim: int = 256
    output_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MultimodalFusion(nn.Module):
    """
    Multimodal fusion module for combining features from different modalities
    """
    
    def __init__(self, config: FusionConfig):
        """
        Initialize multimodal fusion module
        
        Args:
            config: Fusion configuration
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Create modality projection layers
        self.projections = nn.ModuleList()
        for dim in config.input_dims:
            self.projections.append(nn.Linear(dim, config.hidden_dim))
        
        # Create fusion layer based on fusion type
        if config.fusion_type == FusionType.CONCAT:
            self.fusion_layer = nn.Linear(len(config.input_dims) * config.hidden_dim, config.output_dim)
        elif config.fusion_type == FusionType.ADD:
            self.fusion_layer = nn.Linear(config.hidden_dim, config.output_dim)
        elif config.fusion_type == FusionType.MULTIPLY:
            self.fusion_layer = nn.Linear(config.hidden_dim, config.output_dim)
        elif config.fusion_type == FusionType.BILINEAR:
            self.fusion_layer = nn.Bilinear(config.hidden_dim, config.hidden_dim, config.output_dim)
        elif config.fusion_type == FusionType.ATTENTION:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            self.fusion_layer = nn.Linear(config.hidden_dim, config.output_dim)
        elif config.fusion_type == FusionType.CROSS_ATTENTION:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            self.fusion_layer = nn.Linear(config.hidden_dim, config.output_dim)
        elif config.fusion_type == FusionType.GATE:
            self.gate = nn.ModuleList()
            for _ in range(len(config.input_dims)):
                self.gate.append(nn.Linear(config.hidden_dim, 1))
            self.fusion_layer = nn.Linear(config.hidden_dim, config.output_dim)
        elif config.fusion_type == FusionType.DENSE:
            self.fusion_layer = nn.Sequential(
                nn.Linear(len(config.input_dims) * config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.output_dim)
            )
        
        # Create output layer
        self.output_layer = nn.Linear(config.output_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.output_dim)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of multimodal fusion module
        
        Args:
            features: List of feature tensors from different modalities
        
        Returns:
            Fused feature tensor
        """
        # Project features to common hidden dimension
        projected_features = []
        for i, feature in enumerate(features):
            projected = self.projections[i](feature)
            projected = F.relu(projected)
            projected = self.dropout(projected)
            projected_features.append(projected)
        
        # Apply fusion based on fusion type
        if self.config.fusion_type == FusionType.CONCAT:
            fused = torch.cat(projected_features, dim=-1)
            fused = self.fusion_layer(fused)
        elif self.config.fusion_type == FusionType.ADD:
            fused = torch.stack(projected_features, dim=0).sum(dim=0)
            fused = self.fusion_layer(fused)
        elif self.config.fusion_type == FusionType.MULTIPLY:
            fused = torch.stack(projected_features, dim=0).prod(dim=0)
            fused = self.fusion_layer(fused)
        elif self.config.fusion_type == FusionType.BILINEAR:
            # Only supports two modalities for bilinear fusion
            if len(projected_features) != 2:
                raise ValueError("Bilinear fusion only supports two modalities")
            fused = self.fusion_layer(projected_features[0], projected_features[1])
        elif self.config.fusion_type == FusionType.ATTENTION:
            # Reshape for attention
            x = torch.stack(projected_features, dim=0)
            x, _ = self.attention(x, x, x)
            fused = x.mean(dim=0)
            fused = self.fusion_layer(fused)
        elif self.config.fusion_type == FusionType.CROSS_ATTENTION:
            # Cross attention between modalities
            # This is a simplified implementation
            x = projected_features[0].unsqueeze(0)
            context = torch.stack(projected_features[1:], dim=0)
            x, _ = self.cross_attention(x, context, context)
            fused = x.squeeze(0)
            fused = self.fusion_layer(fused)
        elif self.config.fusion_type == FusionType.GATE:
            # Gated fusion
            gates = []
            for i, feature in enumerate(projected_features):
                gate = torch.sigmoid(self.gate[i](feature))
                gates.append(gate)
            
            # Apply gates
            gated_features = []
            for i, (feature, gate) in enumerate(zip(projected_features, gates)):
                gated = feature * gate
                gated_features.append(gated)
            
            fused = torch.stack(gated_features, dim=0).sum(dim=0)
            fused = self.fusion_layer(fused)
        elif self.config.fusion_type == FusionType.DENSE:
            fused = torch.cat(projected_features, dim=-1)
            fused = self.fusion_layer(fused)
        else:
            # Default to concatenation
            fused = torch.cat(projected_features, dim=-1)
            fused = self.fusion_layer(fused)
        
        # Apply output layer
        fused = self.output_layer(fused)
        fused = self.layer_norm(fused)
        
        return fused


def fuse_features(
    features: List[Union[torch.Tensor, np.ndarray]],
    fusion_type: Union[str, FusionType],
    **kwargs
) -> Union[torch.Tensor, np.ndarray]:
    """
    Fuse features from different modalities
    
    Args:
        features: List of feature tensors/arrays from different modalities
        fusion_type: Type of fusion to use
        **kwargs: Additional fusion parameters
    
    Returns:
        Fused feature tensor/array
    """
    # Convert fusion_type string to FusionType enum
    if isinstance(fusion_type, str):
        fusion_type = FusionType(fusion_type)
    
    # Check if features are numpy arrays or torch tensors
    is_numpy = isinstance(features[0], np.ndarray)
    
    # Convert numpy arrays to torch tensors if needed
    if is_numpy:
        features = [torch.tensor(feature) for feature in features]
    
    # Get input dimensions
    input_dims = [feature.shape[-1] for feature in features]
    
    # Create fusion config
    config = FusionConfig(
        fusion_type=fusion_type,
        input_dims=input_dims,
        **kwargs
    )
    
    # Create fusion module
    fusion = MultimodalFusion(config)
    
    # Fuse features
    fused = fusion(features)
    
    # Convert back to numpy array if needed
    if is_numpy:
        fused = fused.detach().numpy()
    
    return fused


def get_fusion_model(
    fusion_type: Union[str, FusionType],
    input_dims: List[int],
    **kwargs
) -> MultimodalFusion:
    """
    Get fusion model
    
    Args:
        fusion_type: Type of fusion to use
        input_dims: List of input dimensions for each modality
        **kwargs: Additional fusion parameters
    
    Returns:
        MultimodalFusion model
    """
    # Convert fusion_type string to FusionType enum
    if isinstance(fusion_type, str):
        fusion_type = FusionType(fusion_type)
    
    # Create fusion config
    config = FusionConfig(
        fusion_type=fusion_type,
        input_dims=input_dims,
        **kwargs
    )
    
    # Create fusion module
    fusion = MultimodalFusion(config)
    
    return fusion


class CLIPFusion(nn.Module):
    """
    CLIP-based multimodal fusion module
    """
    
    def __init__(self, vision_dim: int = 512, text_dim: int = 512, output_dim: int = 256):
        """
        Initialize CLIP fusion module
        
        Args:
            vision_dim: Vision feature dimension
            text_dim: Text feature dimension
            output_dim: Output dimension after fusion
        """
        super().__init__()
        
        # Projection layers
        self.vision_proj = nn.Linear(vision_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CLIP fusion module
        
        Args:
            vision_features: Vision features
            text_features: Text features
        
        Returns:
            Fused feature tensor
        """
        # Project features
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)
        
        # Normalize features
        vision_proj = F.normalize(vision_proj, dim=-1)
        text_proj = F.normalize(text_proj, dim=-1)
        
        # Fuse features
        fused = torch.cat([vision_proj, text_proj], dim=-1)
        fused = self.fusion(fused)
        fused = self.layer_norm(fused)
        
        return fused


class VisionLanguageFusion(nn.Module):
    """
    Vision-language fusion module
    """
    
    def __init__(self, vision_dim: int, language_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        """
        Initialize vision-language fusion module
        
        Args:
            vision_dim: Vision feature dimension
            language_dim: Language feature dimension
            hidden_dim: Hidden dimension for fusion
            output_dim: Output dimension after fusion
        """
        super().__init__()
        
        # Projection layers
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        # Cross attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, vision_features: torch.Tensor, language_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of vision-language fusion module
        
        Args:
            vision_features: Vision features
            language_features: Language features
        
        Returns:
            Fused feature tensor
        """
        # Project features
        vision_proj = self.vision_proj(vision_features)
        language_proj = self.language_proj(language_features)
        
        # Reshape for attention
        vision_proj = vision_proj.unsqueeze(0)
        language_proj = language_proj.unsqueeze(0)
        
        # Cross attention
        attended, _ = self.cross_attention(vision_proj, language_proj, language_proj)
        attended = attended.squeeze(0)
        
        # Fuse features
        fused = torch.cat([vision_proj.squeeze(0), attended], dim=-1)
        fused = self.fusion(fused)
        fused = self.layer_norm(fused)
        
        return fused


def get_clip_fusion(vision_dim: int = 512, text_dim: int = 512, output_dim: int = 256) -> CLIPFusion:
    """
    Get CLIP fusion model
    
    Args:
        vision_dim: Vision feature dimension
        text_dim: Text feature dimension
        output_dim: Output dimension after fusion
    
    Returns:
        CLIPFusion model
    """
    return CLIPFusion(vision_dim, text_dim, output_dim)


def get_vision_language_fusion(
    vision_dim: int,
    language_dim: int,
    hidden_dim: int = 256,
    output_dim: int = 128
) -> VisionLanguageFusion:
    """
    Get vision-language fusion model
    
    Args:
        vision_dim: Vision feature dimension
        language_dim: Language feature dimension
        hidden_dim: Hidden dimension for fusion
        output_dim: Output dimension after fusion
    
    Returns:
        VisionLanguageFusion model
    """
    return VisionLanguageFusion(vision_dim, language_dim, hidden_dim, output_dim)


def fuse_clip_features(
    vision_features: Union[torch.Tensor, np.ndarray],
    text_features: Union[torch.Tensor, np.ndarray],
    output_dim: int = 256
) -> Union[torch.Tensor, np.ndarray]:
    """
    Fuse CLIP features
    
    Args:
        vision_features: Vision features
        text_features: Text features
        output_dim: Output dimension after fusion
    
    Returns:
        Fused feature tensor/array
    """
    # Check if features are numpy arrays or torch tensors
    is_numpy = isinstance(vision_features, np.ndarray)
    
    # Convert numpy arrays to torch tensors if needed
    if is_numpy:
        vision_features = torch.tensor(vision_features)
        text_features = torch.tensor(text_features)
    
    # Create CLIP fusion model
    fusion = CLIPFusion(
        vision_dim=vision_features.shape[-1],
        text_dim=text_features.shape[-1],
        output_dim=output_dim
    )
    
    # Fuse features
    fused = fusion(vision_features, text_features)
    
    # Convert back to numpy array if needed
    if is_numpy:
        fused = fused.detach().numpy()
    
    return fused
