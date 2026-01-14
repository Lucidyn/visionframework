"""
Pose Estimator

Provides pose estimation for human/object keypoint detection.
"""

from typing import List, Optional, Any, Dict, Tuple
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from .feature_extractor import FeatureExtractor
from ...data.pose import Pose, KeyPoint
from ...utils.logger import get_logger

logger = get_logger(__name__)


class PoseEstimator(FeatureExtractor):
    """
    Pose estimator for human/object pose detection using YOLO Pose.
    
    Detects keypoints (joints) and provides skeleton connections.
    """
    
    # COCO keypoint names (17 keypoints)
    COCO_KEYPOINT_NAMES: List[str] = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    def __init__(self, model_name: str = "yolov8n-pose.pt", 
                 device: str = "cpu",
                 conf_threshold: float = 0.25,
                 keypoint_threshold: float = 0.5):
        """
        Initialize pose estimator.
        
        Args:
            model_name: Model path or identifier
            device: Device to run on ("cpu", "cuda", etc.)
            conf_threshold: Confidence threshold for pose detection
            keypoint_threshold: Confidence threshold for individual keypoints
        """
        super().__init__(model_name, device)
        self.conf_threshold = conf_threshold
        self.keypoint_threshold = keypoint_threshold
        self.keypoint_names = self.COCO_KEYPOINT_NAMES
    
    def initialize(self) -> None:
        """Initialize the pose estimator model."""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. "
                            "Install with: pip install 'visionframework[yolo]'")
        
        try:
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            self._initialized = True
            logger.info(f"Pose estimator initialized with {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize pose estimator: {e}", exc_info=True)
            raise RuntimeError(f"Pose estimator initialization failed: {e}")
    
    def extract(self, image: np.ndarray) -> List[Pose]:
        """
        Extract poses from image.
        
        Args:
            image: Input image (BGR)
        
        Returns:
            List of Pose objects
        """
        return self.process(image)
    
    def process(self, image: np.ndarray) -> List[Pose]:
        """
        Estimate poses in image.
        
        Args:
            image: Input image in BGR format (H, W, 3)
        
        Returns:
            List[Pose]: Detected poses with keypoints
        """
        if not self.is_initialized():
            self.initialize()
        
        poses: List[Pose] = []
        
        try:
            # Run YOLO Pose inference
            results = self.model(
                image,
                conf=self.conf_threshold,
                verbose=False
            )
            
            # Process results
            for result in results:
                if result.keypoints is not None:
                    boxes = result.boxes
                    keypoints_data = result.keypoints
                    
                    for i in range(len(boxes)):
                        # Extract bounding box
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        
                        # Extract keypoints [num_keypoints, 3] = (x, y, confidence)
                        keypoints = keypoints_data.data[i].cpu().numpy()
                        
                        keypoint_list: List[KeyPoint] = []
                        for j, kp in enumerate(keypoints):
                            if len(kp) >= 3 and kp[2] >= self.keypoint_threshold:
                                kp_name = (self.keypoint_names[j] 
                                          if j < len(self.keypoint_names) 
                                          else f"keypoint_{j}")
                                keypoint = KeyPoint(
                                    x=float(kp[0]),
                                    y=float(kp[1]),
                                    confidence=float(kp[2]),
                                    keypoint_id=j,
                                    keypoint_name=kp_name
                                )
                                keypoint_list.append(keypoint)
                        
                        # Create Pose object
                        pose = Pose(
                            bbox=(float(box[0]), float(box[1]), 
                                  float(box[2]), float(box[3])),
                            keypoints=keypoint_list,
                            confidence=conf,
                            pose_id=i
                        )
                        poses.append(pose)
            
            return poses
        except Exception as e:
            logger.error(f"Pose estimation error: {e}", exc_info=True)
            return []
    
    def estimate(self, image: np.ndarray) -> List[Pose]:
        """Alias for process method."""
        return self.process(image)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_initialized():
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "model_name": self.model_name,
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "keypoint_threshold": self.keypoint_threshold,
            "num_keypoints": len(self.keypoint_names),
        }
    
    def _move_to_device(self, device: str) -> None:
        """Move model to device."""
        if self.model is not None:
            self.model.to(device)
