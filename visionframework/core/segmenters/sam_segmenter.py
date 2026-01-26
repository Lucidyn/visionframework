#!/usr/bin/env python3
"""
SAM (Segment Anything Model) segmenter implementation

This module provides a wrapper around the Segment Anything Model (SAM) for image segmentation.
It supports both automatic segmentation and interactive segmentation with prompts.
"""

from typing import List, Optional, Tuple, Any, Dict
import numpy as np
from ..base import BaseModule
from ...data.detection import Detection
from ...utils.monitoring.logger import get_logger
from ...utils.io.config_models import ModelCache

logger = get_logger(__name__)


class SAMSegmenter(BaseModule):
    """
    Segment Anything Model (SAM) segmenter
    
    This class provides image segmentation functionality using Meta's SAM model.
    It supports both automatic segmentation and interactive segmentation with prompts.
    
    Example:
        ```python
        # Automatic segmentation
        sam = SAMSegmenter({
            "model_type": "vit_b",
            "device": "cuda",
            "use_fp16": True
        })
        sam.initialize()
        masks = sam.automatic_segment(image)
        
        # Interactive segmentation with points
        masks = sam.segment_with_points(image, points=[(100, 150)], labels=[1])
        
        # Interactive segmentation with boxes
        masks = sam.segment_with_boxes(image, boxes=[(50, 50, 200, 200)])
        ```
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SAM segmenter
        
        Args:
            config: Configuration dictionary with keys:
                - model_type: SAM model type, one of: 'vit_h', 'vit_l', 'vit_b' (default: 'vit_b')
                - model_path: Path to SAM model checkpoint (default: None, downloads automatically)
                - device: Device to use, one of: 'cpu', 'cuda', 'mps' (default: 'cpu')
                - use_fp16: Whether to use FP16 precision (default: True for CUDA, False for CPU)
                - automatic_threshold: Threshold for automatic segmentation quality (default: 0.8)
                - max_masks: Maximum number of masks to return for automatic segmentation (default: 100)
        """
        super().__init__(config)
        self.model_type: str = self.config.get("model_type", "vit_b")
        self.model_path: Optional[str] = self.config.get("model_path")
        self.device: str = self.config.get("device", "cpu")
        self.use_fp16: bool = self.config.get("use_fp16", self.device == "cuda")
        self.automatic_threshold: float = self.config.get("automatic_threshold", 0.8)
        self.max_masks: int = self.config.get("max_masks", 100)
        
        # Model and processor
        self.model = None
        self.processor = None
        self.predictor = None
        
        # Model cache key
        self._model_cache_key: Optional[str] = None
    
    def initialize(self) -> bool:
        """
        Initialize SAM model and processor
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Lazy import to avoid heavy dependencies on startup
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            import torch
            
            # Build cache key
            key = f"sam:{self.model_type}:fp16={self.use_fp16}"
            
            def loader():
                """Load SAM model"""
                # Get model checkpoint path
                model_checkpoint = self.model_path
                if not model_checkpoint:
                    # Download from SAM GitHub if not provided
                    from huggingface_hub import hf_hub_download
                    model_checkpoint = hf_hub_download(
                        repo_id="facebook/sam",
                        filename=f"sam_{self.model_type}.pth",
                        local_dir="./models"
                    )
                
                # Load model
                sam = sam_model_registry[self.model_type](checkpoint=model_checkpoint)
                
                # Set to eval mode
                sam.eval()
                
                return sam
            
            # Load or get cached model
            self.model = ModelCache.get_model(key, loader)
            self._model_cache_key = key
            
            # Move model to device
            self.model.to(self.device)
            
            # Set up automatic mask generator and predictor
            self.predictor = SamPredictor(self.model)
            self.automatic_mask_generator = SamAutomaticMaskGenerator(
                self.model,
                points_per_side=32,
                pred_iou_thresh=self.automatic_threshold,
                stability_score_thresh=0.95,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100
            )
            
            # Enable FP16 if requested
            if self.use_fp16 and self.device == "cuda":
                self.model.half()
            
            self.is_initialized = True
            logger.info(f"SAM segmenter initialized with model: {self.model_type}")
            return True
            
        except ImportError as e:
            logger.error(f"Missing dependencies for SAM segmenter: {e}")
            logger.info("Install with: pip install 'segment-anything'")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize SAM segmenter: {e}", exc_info=True)
            return False
    
    def automatic_segment(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform automatic segmentation on an image
        
        Args:
            image: Input image in BGR format
        
        Returns:
            List[Dict[str, Any]]: List of masks with metadata
                Each mask dict contains:
                - segmentation: np.ndarray (bool) - Binary mask
                - area: int - Mask area in pixels
                - bbox: Tuple[int, int, int, int] - Bounding box (x, y, w, h)
                - predicted_iou: float - Model's prediction of mask quality
                - point_coords: List[List[float]] - The point coordinates used to generate the mask
                - stability_score: float - Stability of the mask
                - crop_box: List[int] - The crop of the image used to generate the mask
        """
        if not self.is_initialized:
            if not self.initialize():
                return []
        
        try:
            # Convert BGR to RGB (SAM expects RGB)
            image_rgb = image[..., ::-1]
            
            # Generate masks
            masks = self.automatic_mask_generator.generate(image_rgb)
            
            # Filter and sort masks
            masks = sorted(masks, key=lambda x: x["area"], reverse=True)[:self.max_masks]
            
            return masks
            
        except Exception as e:
            logger.error(f"Error during automatic segmentation: {e}", exc_info=True)
            return []
    
    def segment_with_points(
        self,
        image: np.ndarray,
        points: List[Tuple[int, int]],
        labels: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Segment image with point prompts
        
        Args:
            image: Input image in BGR format
            points: List of (x, y) coordinates
            labels: List of labels (1 for foreground, 0 for background)
        
        Returns:
            List[Dict[str, Any]]: List of masks with metadata
        """
        if not self.is_initialized:
            if not self.initialize():
                return []
        
        try:
            # Convert BGR to RGB
            image_rgb = image[..., ::-1]
            
            # Set image for predictor
            self.predictor.set_image(image_rgb)
            
            # Convert points to numpy array
            point_coords = np.array(points)
            point_labels = np.array(labels)
            
            # Predict masks
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
            
            # Prepare results
            results = []
            for i, (mask, score) in enumerate(zip(masks, scores)):
                results.append({
                    "segmentation": mask,
                    "score": float(score),
                    "logits": logits[i],
                    "type": "point_prompt"
                })
            
            # Sort by score
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during point-based segmentation: {e}", exc_info=True)
            return []
    
    def segment_with_boxes(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[Dict[str, Any]]:
        """
        Segment image with bounding box prompts
        
        Args:
            image: Input image in BGR format
            boxes: List of (x1, y1, x2, y2) boxes
        
        Returns:
            List[Dict[str, Any]]: List of masks with metadata
        """
        if not self.is_initialized:
            if not self.initialize():
                return []
        
        try:
            # Convert BGR to RGB
            image_rgb = image[..., ::-1]
            
            # Set image for predictor
            self.predictor.set_image(image_rgb)
            
            results = []
            
            # Process each box
            for box in boxes:
                # Convert box to numpy array
                input_box = np.array(box)
                
                # Predict masks
                masks, scores, logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=True
                )
                
                # Prepare results
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    results.append({
                        "segmentation": mask,
                        "score": float(score),
                        "logits": logits[i],
                        "box": box,
                        "type": "box_prompt"
                    })
            
            # Sort by score
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during box-based segmentation: {e}", exc_info=True)
            return []
    
    def segment_detections(
        self,
        image: np.ndarray,
        detections: List[Detection]
    ) -> List[Detection]:
        """
        Segment objects based on detection boxes
        
        Args:
            image: Input image in BGR format
            detections: List of Detection objects
        
        Returns:
            List[Detection]: Detections with added segmentation masks
        """
        if not detections:
            return detections
        
        # Extract boxes from detections
        boxes = [det.bbox for det in detections]
        
        # Get masks for boxes
        mask_results = self.segment_with_boxes(image, boxes)
        
        # Assign masks to detections
        for i, det in enumerate(detections):
            if i < len(mask_results):
                det.mask = mask_results[i]["segmentation"]
                det.mask_score = mask_results[i]["score"]
        
        return detections
    
    def process(
        self,
        image: np.ndarray,
        detections: Optional[List[Detection]] = None,
        **kwargs
    ) -> Any:
        """
        Process input data (required by BaseModule)
        
        Args:
            image: Input image in BGR format
            detections: Optional list of Detection objects
            **kwargs: Additional keyword arguments
            
        Returns:
            Processed results based on input
        """
        if detections:
            return self.segment_detections(image, detections)
        else:
            return self.automatic_segment(image)
    
    def process_batch(
        self,
        images: List[np.ndarray],
        detections_list: Optional[List[List[Detection]]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Process multiple images in batch mode
        
        Args:
            images: List of input images in BGR format
            detections_list: Optional list of detections per image
        
        Returns:
            List[List[Dict[str, Any]]]: List of mask results per image
        """
        results = []
        
        for i, image in enumerate(images):
            if detections_list and i < len(detections_list):
                # Segment based on detections
                masks = self.segment_detections(image, detections_list[i])
                results.append([{
                    "segmentation": det.mask,
                    "detection": det
                } for det in masks if det.mask is not None])
            else:
                # Automatic segmentation
                masks = self.automatic_segment(image)
                results.append(masks)
        
        return results
    
    def cleanup(self) -> None:
        """
        Cleanup SAM model and resources
        """
        try:
            if self._model_cache_key:
                ModelCache.release_model(self._model_cache_key)
                self._model_cache_key = None
            
            # Clear references
            self.model = None
            self.processor = None
            self.predictor = None
            self.automatic_mask_generator = None
        except Exception as e:
            logger.warning(f"Error during SAM cleanup: {e}")
        finally:
            self.is_initialized = False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "status": "initialized" if self.is_initialized else "not_initialized",
            "model_type": self.model_type,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "automatic_threshold": self.automatic_threshold,
            "max_masks": self.max_masks
        }
