"""
Model fine-tuning functionality

This module provides functionality for fine-tuning models on custom datasets.
"""

import os
import time
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


# ---------------------------------------------------------------------------
# Lightweight built-in LoRA adapter (used when peft is not installed)
# ---------------------------------------------------------------------------

class _LoRALinear(nn.Module):
    """Drop-in LoRA wrapper for nn.Linear."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = linear
        self.rank = rank
        self.scale = alpha / rank
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base_out + lora_out


def _inject_lora_adapters(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> nn.Module:
    """Replace all nn.Linear layers with _LoRALinear wrappers (in-place)."""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            setattr(model, name, _LoRALinear(module, rank=rank, alpha=alpha))
        else:
            _inject_lora_adapters(module, rank=rank, alpha=alpha)
    return model


class FineTuningStrategy(Enum):
    """Fine-tuning strategy enum"""
    FULL = "full"
    FREEZE = "freeze"
    LORA = "lora"
    QLORA = "qlora"


class ModelType(Enum):
    """Model type enum"""
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    POSE = "pose"
    REID = "reid"
    CLIP = "clip"
    FACE = "face"
    DETR = "detr"
    RFDETR = "rfdetr"
    SAM = "sam"


@dataclass
class FineTuningConfig:
    """
    Fine-tuning configuration class
    
    Attributes:
        strategy: Fine-tuning strategy
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        optimizer: Optimizer type
        scheduler: Learning rate scheduler
        device: Device to use
        verbose: Verbosity level
        save_best: Save best model
        save_dir: Directory to save models
        early_stopping: Early stopping patience
        grad_accumulation_steps: Gradient accumulation steps
        mixed_precision: Use mixed precision training
    """
    strategy: FineTuningStrategy = FineTuningStrategy.FREEZE
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = True
    save_best: bool = True
    save_dir: str = "./fine_tuned_models"
    early_stopping: int = 5
    grad_accumulation_steps: int = 1
    mixed_precision: bool = True


class ModelFineTuner:
    """
    Model fine-tuner class for fine-tuning models on custom datasets
    """
    
    def __init__(self, config: FineTuningConfig):
        """
        Initialize model fine-tuner
        
        Args:
            config: Fine-tuning configuration
        """
        self.config = config
        self.device = torch.device(self.config.device)
        
        # Create save directory if it doesn't exist
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Initialize variables
        self.best_loss = float('inf')
        self.early_stopping_counter = 0
    
    def fine_tune(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        model_type: Optional[ModelType] = None
    ) -> Dict[str, Any]:
        """
        Fine-tune model on dataset
        
        Args:
            model: Model to fine-tune
            train_dataset: Training dataset
            val_dataset: Validation dataset
            model_type: Type of model
        
        Returns:
            Dictionary with training results
        """
        # Prepare model based on strategy
        model = self._prepare_model(model, self.config.strategy, model_type)
        model.to(self.device)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4
            )
        
        # Create optimizer
        optimizer = self._create_optimizer(model)
        
        # Create scheduler
        scheduler = self._create_scheduler(optimizer, len(train_loader))
        
        # Create scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision and self.device.type == "cuda" else None
        
        # Training loop
        results = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "best_model_path": None
        }
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            if self.config.verbose:
                print(f"Epoch {epoch+1}/{self.config.epochs}")
                print("-" * 50)
            
            # Train for one epoch
            train_loss, train_acc = self._train_epoch(
                model, train_loader, optimizer, scheduler, scaler
            )
            
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            
            if self.config.verbose:
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            
            # Validate if validation dataset is provided
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(model, val_loader)
                results["val_loss"].append(val_loss)
                results["val_acc"].append(val_acc)
                
                if self.config.verbose:
                    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                
                # Check if this is the best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.early_stopping_counter = 0
                    
                    if self.config.save_best:
                        best_model_path = os.path.join(
                            self.config.save_dir,
                            f"best_model_epoch_{epoch+1}.pt"
                        )
                        torch.save(model.state_dict(), best_model_path)
                        results["best_model_path"] = best_model_path
                        if self.config.verbose:
                            print(f"Saved best model to {best_model_path}")
                else:
                    self.early_stopping_counter += 1
                    if self.config.verbose:
                        print(f"Early stopping counter: {self.early_stopping_counter}/{self.config.early_stopping}")
                    
                    # Early stopping
                    if self.early_stopping_counter >= self.config.early_stopping:
                        if self.config.verbose:
                            print("Early stopping triggered")
                        break
            
            print()
        
        # Save final model
        final_model_path = os.path.join(
            self.config.save_dir,
            "final_model.pt"
        )
        torch.save(model.state_dict(), final_model_path)
        results["final_model_path"] = final_model_path
        
        if self.config.verbose:
            print(f"Training completed in {time.time() - start_time:.2f} seconds")
            print(f"Final model saved to {final_model_path}")
        
        return results
    
    def _prepare_model(
        self,
        model: nn.Module,
        strategy: FineTuningStrategy,
        model_type: Optional[ModelType] = None
    ) -> nn.Module:
        """
        Prepare model based on fine-tuning strategy
        
        Args:
            model: Model to prepare
            strategy: Fine-tuning strategy
            model_type: Type of model
        
        Returns:
            Prepared model
        """
        if strategy == FineTuningStrategy.FREEZE:
            # Freeze all layers except the last one
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze the last layer based on model type
            if model_type == ModelType.DETECTION:
                # For detection models, unfreeze the classification head
                if hasattr(model, 'model') and hasattr(model.model, 'head'):
                    for param in model.model.head.parameters():
                        param.requires_grad = True
            elif model_type == ModelType.CLIP:
                # For CLIP models, unfreeze the projection layers
                if hasattr(model, 'visual_projection'):
                    for param in model.visual_projection.parameters():
                        param.requires_grad = True
                if hasattr(model, 'text_projection'):
                    for param in model.text_projection.parameters():
                        param.requires_grad = True
            else:
                # For other models, try to unfreeze the last layer
                if hasattr(model, 'fc'):
                    for param in model.fc.parameters():
                        param.requires_grad = True
                elif hasattr(model, 'classifier'):
                    for param in model.classifier.parameters():
                        param.requires_grad = True
        elif strategy == FineTuningStrategy.FULL:
            # Unfreeze all layers
            for param in model.parameters():
                param.requires_grad = True
        elif strategy == FineTuningStrategy.LORA:
            # Freeze all base parameters first
            for param in model.parameters():
                param.requires_grad = False

            # Try peft-based LoRA; fall back to lightweight built-in LoRA
            try:
                from peft import get_peft_model, LoraConfig, TaskType  # type: ignore
                lora_cfg = LoraConfig(
                    r=self.config.lora_rank if hasattr(self.config, "lora_rank") else 8,
                    lora_alpha=self.config.lora_alpha if hasattr(self.config, "lora_alpha") else 16,
                    lora_dropout=self.config.lora_dropout if hasattr(self.config, "lora_dropout") else 0.1,
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                model = get_peft_model(model, lora_cfg)
            except (ImportError, Exception):
                # Fallback: inject lightweight LoRA adapters into Linear layers
                model = _inject_lora_adapters(
                    model,
                    rank=getattr(self.config, "lora_rank", 8),
                    alpha=getattr(self.config, "lora_alpha", 16),
                )

        elif strategy == FineTuningStrategy.QLORA:
            # Freeze all base parameters first
            for param in model.parameters():
                param.requires_grad = False

            # Try bitsandbytes + peft QLoRA; fall back to LoRA without quantization
            try:
                import bitsandbytes as bnb  # type: ignore
                from peft import get_peft_model, LoraConfig, TaskType  # type: ignore
                lora_cfg = LoraConfig(
                    r=self.config.lora_rank if hasattr(self.config, "lora_rank") else 4,
                    lora_alpha=self.config.lora_alpha if hasattr(self.config, "lora_alpha") else 8,
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                model = get_peft_model(model, lora_cfg)
            except (ImportError, Exception):
                # Fallback: same lightweight LoRA adapters (quantization skipped)
                model = _inject_lora_adapters(
                    model,
                    rank=getattr(self.config, "lora_rank", 4),
                    alpha=getattr(self.config, "lora_alpha", 8),
                )
        
        return model
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """
        Create optimizer based on configuration
        
        Args:
            model: Model to optimize
        
        Returns:
            Optimizer instance
        """
        # Get parameters that require grad
        params = [p for p in model.parameters() if p.requires_grad]
        
        if self.config.optimizer == "adam":
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "rmsprop":
            return optim.RMSprop(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(
        self,
        optimizer: optim.Optimizer,
        train_steps: int
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler based on configuration
        
        Args:
            optimizer: Optimizer to schedule
            train_steps: Number of training steps per epoch
        
        Returns:
            Scheduler instance
        """
        if self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs * train_steps
            )
        elif self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=3,
                gamma=0.1
            )
        elif self.config.scheduler == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=2
            )
        elif self.config.scheduler == "one_cycle":
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=train_steps
            )
        elif self.config.scheduler is None:
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler]
    ) -> Tuple[float, float]:
        """
        Train model for one epoch
        
        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            scaler: Gradient scaler for mixed precision
        
        Returns:
            Tuple of (average loss, average accuracy)
        """
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for step, batch in enumerate(train_loader):
            # Move batch to device
            if isinstance(batch, Tuple) and len(batch) == 2:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            else:
                # Handle different batch formats
                # This would be model-specific
                inputs = batch["image"].to(self.device)
                targets = batch["label"].to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = self._calculate_loss(outputs, targets)
            else:
                outputs = model(inputs)
                loss = self._calculate_loss(outputs, targets)
            
            # Backward pass with mixed precision
            if scaler is not None:
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (step + 1) % self.config.grad_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                
                # Gradient accumulation
                if (step + 1) % self.config.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Update scheduler
            if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            
            # Calculate metrics
            total_loss += loss.item() * inputs.size(0)
            total_correct += self._calculate_correct(outputs, targets)
            total_samples += inputs.size(0)
            
            # Print progress
            if self.config.verbose and (step + 1) % 10 == 0:
                print(f"Step {step+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """
        Validate model for one epoch
        
        Args:
            model: Model to validate
            val_loader: Validation data loader
        
        Returns:
            Tuple of (average loss, average accuracy)
        """
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                if isinstance(batch, Tuple) and len(batch) == 2:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    # Handle different batch formats
                    # This would be model-specific
                    inputs = batch["image"].to(self.device)
                    targets = batch["label"].to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                loss = self._calculate_loss(outputs, targets)
                
                # Calculate metrics
                total_loss += loss.item() * inputs.size(0)
                total_correct += self._calculate_correct(outputs, targets)
                total_samples += inputs.size(0)
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def _calculate_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        """
        Calculate loss based on model outputs and targets
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
        
        Returns:
            Loss tensor
        """
        # This would be model-specific
        # For classification models
        if isinstance(outputs, torch.Tensor) and outputs.dim() == 2:
            return nn.CrossEntropyLoss()(outputs, targets)
        # For detection models
        elif isinstance(outputs, Dict):
            # Detection models typically return a dict with loss
            if "loss" in outputs:
                return outputs["loss"]
            else:
                # Calculate loss from boxes and labels
                return nn.MSELoss()(outputs["boxes"], targets["boxes"])
        else:
            # Default loss function
            return nn.MSELoss()(outputs, targets)
    
    def _calculate_correct(self, outputs: Any, targets: Any) -> int:
        """
        Calculate number of correct predictions
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
        
        Returns:
            Number of correct predictions
        """
        # This would be model-specific
        # For classification models
        if isinstance(outputs, torch.Tensor) and outputs.dim() == 2:
            _, predicted = torch.max(outputs, 1)
            return (predicted == targets).sum().item()
        # For other models
        else:
            # This would be model-specific
            return 0


def fine_tune_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    config: Optional[FineTuningConfig] = None,
    model_type: Optional[Union[str, ModelType]] = None
) -> Dict[str, Any]:
    """
    Fine-tune model on dataset
    
    Args:
        model: Model to fine-tune
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Fine-tuning configuration
        model_type: Type of model
    
    Returns:
        Dictionary with training results
    """
    # Create default config if not provided
    if config is None:
        config = FineTuningConfig()
    
    # Convert model_type string to ModelType enum
    if isinstance(model_type, str):
        model_type = ModelType(model_type)
    
    # Create fine tuner
    tuner = ModelFineTuner(config)
    
    # Fine-tune model
    return tuner.fine_tune(model, train_dataset, val_dataset, model_type)


def get_default_transforms() -> transforms.Compose:
    """
    Get default transforms for fine-tuning
    
    Returns:
        Transform composition
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Get validation transforms for fine-tuning
    
    Returns:
        Transform composition
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def create_image_folder_dataset(
    data_dir: str,
    train_transforms: Optional[transforms.Compose] = None,
    val_transforms: Optional[transforms.Compose] = None
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Create image folder dataset for fine-tuning
    
    Args:
        data_dir: Directory with images organized in subdirectories
        train_transforms: Transforms for training
        val_transforms: Transforms for validation
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Use default transforms if not provided
    if train_transforms is None:
        train_transforms = get_default_transforms()
    if val_transforms is None:
        val_transforms = get_val_transforms()
    
    # Check if data_dir has train and val subdirectories
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        train_dataset = ImageFolder(train_dir, transform=train_transforms)
        val_dataset = ImageFolder(val_dir, transform=val_transforms)
    else:
        # Use entire dataset for training
        train_dataset = ImageFolder(data_dir, transform=train_transforms)
        val_dataset = None
    
    return train_dataset, val_dataset
