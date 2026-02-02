"""
Model distillation utilities

This module provides tools for knowledge distillation to transfer knowledge from a teacher model to a student model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from torch.utils.data import DataLoader


@dataclass
class DistillationConfig:
    """
    Configuration for model distillation
    
    Attributes:
        temperature: Temperature for softening logits
        alpha: Weight for distillation loss
        student_loss_weight: Weight for student classification loss
        epochs: Number of distillation epochs
        batch_size: Batch size for distillation
        learning_rate: Learning rate for student model
        device: Device to use for distillation
        verbose: Whether to print verbose information
    """
    temperature: float = 3.0
    alpha: float = 0.7
    student_loss_weight: float = 0.3
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = False


def distill_model(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_data: DataLoader,
    config: DistillationConfig
) -> nn.Module:
    """
    Distill knowledge from a teacher model to a student model
    
    Args:
        teacher_model: Teacher model with more knowledge
        student_model: Student model to distill knowledge into
        train_data: Training data loader
        config: Distillation configuration
    
    Returns:
        Distilled student model
    """
    if config.verbose:
        print(f"Distilling knowledge from teacher model to student model")
        print(f"Temperature: {config.temperature}, Alpha: {config.alpha}")
    
    # Move models to device
    teacher_model.to(config.device)
    student_model.to(config.device)
    
    # Set teacher model to evaluation mode
    teacher_model.eval()
    
    # Set student model to training mode
    student_model.train()
    
    # Define loss functions
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction="batchmean")
    
    # Define optimizer
    optimizer = optim.AdamW(student_model.parameters(), lr=config.learning_rate)
    
    # Define scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Distillation loop
    for epoch in range(config.epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        if config.verbose:
            print(f"Epoch {epoch+1}/{config.epochs}")
        
        for batch_idx, (images, targets) in enumerate(train_data):
            # Move data to device
            images = images.to(config.device)
            targets = targets.to(config.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            
            # Get student predictions
            student_logits = student_model(images)
            
            # Calculate distillation loss
            loss_kd = criterion_kd(
                nn.functional.log_softmax(student_logits / config.temperature, dim=1),
                nn.functional.softmax(teacher_logits / config.temperature, dim=1)
            ) * (config.temperature ** 2)
            
            # Calculate student classification loss
            loss_ce = criterion_ce(student_logits, targets)
            
            # Combine losses
            loss = config.alpha * loss_kd + config.student_loss_weight * loss_ce
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if config.verbose and (batch_idx + 1) % 100 == 0:
                print(f"Batch {batch_idx+1}/{len(train_data)}, Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%")
        
        # Update scheduler
        scheduler.step()
        
        # Print epoch summary
        if config.verbose:
            print(f"Epoch {epoch+1}/{config.epochs} completed")
            print(f"Average Loss: {total_loss/len(train_data):.4f}")
            print(f"Accuracy: {100.*correct/total:.2f}%")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    
    return student_model


def get_distilled_model(
    teacher_model_path: str,
    student_model: nn.Module,
    train_data: DataLoader,
    config: DistillationConfig
) -> nn.Module:
    """
    Load a teacher model from file and distill knowledge to a student model
    
    Args:
        teacher_model_path: Path to teacher model file
        student_model: Student model to distill knowledge into
        train_data: Training data loader
        config: Distillation configuration
    
    Returns:
        Distilled student model
    """
    # Load teacher model
    teacher_model = torch.load(teacher_model_path)
    
    # Distill knowledge
    distilled_model = distill_model(teacher_model, student_model, train_data, config)
    
    return distilled_model
