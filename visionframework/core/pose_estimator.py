"""
Pose estimation module for human/object pose detection
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .base import BaseModule
from ..data.pose import Pose, KeyPoint
from ..utils.monitoring.logger import get_logger

logger = get_logger(__name__)

# Lazy import flags
YOLO_POSE_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False

# Import references (will be set on first use)
YOLO = None
mp = None
mp_pose = None
mp_drawing = None


def _ensure_yolo_import():
    """Ensure YOLO is imported"""
    global YOLO, YOLO_POSE_AVAILABLE
    if not YOLO_POSE_AVAILABLE:
        try:
            from ultralytics import YOLO
            YOLO_POSE_AVAILABLE = True
        except ImportError:
            YOLO_POSE_AVAILABLE = False
            YOLO = None


def _ensure_mediapipe_import():
    """Ensure MediaPipe is imported"""
    global mp, mp_pose, mp_drawing, MEDIAPIPE_AVAILABLE
    if not MEDIAPIPE_AVAILABLE:
        try:
            import mediapipe as mp
            MEDIAPIPE_AVAILABLE = True
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
        except ImportError:
            MEDIAPIPE_AVAILABLE = False
            mp = None
            mp_pose = None
            mp_drawing = None


class PoseEstimator(BaseModule):
    """
    Pose estimator for human/object pose detection
    
    This class provides pose estimation functionality using YOLO Pose models.
    It can detect keypoints (joints) and draw skeleton connections for human poses.
    
    Example:
        ```python
        # Initialize pose estimator
        pose_estimator = PoseEstimator({
            "model_path": "yolov8n-pose.pt",
            "conf_threshold": 0.25,
            "keypoint_threshold": 0.5
        })
        pose_estimator.initialize()
        
        # Estimate poses
        poses = pose_estimator.estimate(image)
        for pose in poses:
            print(f"Pose with {len(pose.keypoints)} keypoints")
        ```
    """
    
    # COCO keypoint names (17 keypoints in COCO format)
    COCO_KEYPOINT_NAMES: List[str] = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pose estimator
        
        Args:
            config: Configuration dictionary with keys:
                - model_path: Path to pose estimation model file (default: 'yolov8n-pose.pt')
                  First use will automatically download the model.
                  Available models: yolov8n-pose.pt, yolov8s-pose.pt, yolov8m-pose.pt, etc.
                - model_type: Type of model (default: 'yolo_pose')
                  Supported: 'yolo_pose', 'mediapipe'
                - conf_threshold: Confidence threshold for pose detection between 0.0 and 1.0 (default: 0.25)
                  Poses with confidence below this threshold are filtered out.
                - keypoint_threshold: Confidence threshold for individual keypoints between 0.0 and 1.0 (default: 0.5)
                  Keypoints with confidence below this threshold are not included in results.
                - device: Device to use for inference, one of:
                    - 'cpu': CPU inference (default)
                    - 'cuda': GPU inference (requires CUDA-capable GPU)
                    - 'mps': Apple Silicon GPU (macOS only)
                - keypoint_names: List of keypoint names (default: COCO_KEYPOINT_NAMES)
                  Should match the number of keypoints in the model.
                - min_detection_confidence: Minimum confidence for initial detection (MediaPipe only, default: 0.5)
                - min_tracking_confidence: Minimum confidence for tracking (MediaPipe only, default: 0.5)
        
        Raises:
            ValueError: If configuration is invalid (will be logged as warning)
        """
        super().__init__(config)
        self.model: Optional[Any] = None  # YOLO or MediaPipe model
        self.model_path: str = self.config.get("model_path", "yolov8n-pose.pt")
        self.model_type: str = self.config.get("model_type", "yolo_pose")
        self.conf_threshold: float = float(self.config.get("conf_threshold", 0.25))
        self.keypoint_threshold: float = float(self.config.get("keypoint_threshold", 0.5))
        self.device: str = self.config.get("device", "cpu")
        self.keypoint_names: List[str] = self.config.get("keypoint_names", self.COCO_KEYPOINT_NAMES)
        
        # MediaPipe specific parameters
        self.min_detection_confidence: float = self.config.get("min_detection_confidence", 0.5)
        self.min_tracking_confidence: float = self.config.get("min_tracking_confidence", 0.5)
    
    def initialize(self) -> bool:
        """Initialize the pose estimator model"""
        try:
            if self.model_type == "yolo_pose":
                # Lazy import YOLO
                _ensure_yolo_import()
                if not YOLO_POSE_AVAILABLE:
                    raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
                global YOLO
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
            elif self.model_type == "mediapipe":
                # Lazy import MediaPipe
                _ensure_mediapipe_import()
                if not MEDIAPIPE_AVAILABLE:
                    raise ImportError("mediapipe not installed. Install with: pip install mediapipe")
                global mp_pose
                # Initialize MediaPipe Pose model
                self.model = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}. Supported: 'yolo_pose', 'mediapipe'")
            
            self.is_initialized = True
            logger.info(f"Pose estimator initialized successfully with model: {self.model_type}")
            return True
        except ImportError as e:
            logger.error(f"Missing dependency for pose estimator: {e}", exc_info=True)
            return False
        except ValueError as e:
            logger.error(f"Invalid pose estimator configuration: {e}", exc_info=True)
            return False
        except RuntimeError as e:
            logger.error(f"Failed to initialize pose estimator model: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing pose estimator: {e}", exc_info=True)
            return False
    
    def process(self, image: np.ndarray) -> List[Pose]:
        """
        Estimate poses in image
        
        This method performs pose estimation on the input image, detecting human
        poses and their keypoints (joints). Each detected pose includes a bounding
        box and a list of keypoints with their coordinates and confidence scores.
        
        Args:
            image: Input image in BGR format (OpenCV standard), numpy array with shape (H, W, 3).
                   Data type should be uint8 with values in range [0, 255].
        
        Returns:
            List[Pose]: List of Pose objects, each containing:
                - bbox: Tuple of (x1, y1, x2, y2) coordinates for the person bounding box
                - keypoints: List of KeyPoint objects, each containing:
                    - x, y: Keypoint coordinates
                    - confidence: Keypoint confidence score
                    - keypoint_id: Integer ID of the keypoint
                    - keypoint_name: String name of the keypoint (e.g., "nose", "left_shoulder")
                - confidence: Overall pose confidence score
                - pose_id: Integer identifier for the pose
        
        Raises:
            RuntimeError: If model is not initialized or pose estimation fails
            ValueError: If image format is invalid
        
        Note:
            Only keypoints with confidence >= keypoint_threshold are included in results.
            YOLO Pose models detect 17 keypoints in COCO format.
        
        Example:
            ```python
            pose_estimator = PoseEstimator()
            pose_estimator.initialize()
            poses = pose_estimator.process(image)
            
            for pose in poses:
                print(f"Pose detected with {len(pose.keypoints)} visible keypoints")
                for kp in pose.keypoints:
                    print(f"  {kp.keypoint_name}: ({kp.x}, {kp.y}), conf={kp.confidence:.2f}")
            ```
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("Pose estimator not initialized")
                return []
        
        poses: List[Pose] = []
        
        try:
            if self.model_type == "yolo_pose":
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
                            
                            # Extract keypoints
                            # Shape: [num_keypoints, 3] where 3 = (x, y, confidence)
                            keypoints = keypoints_data.data[i].cpu().numpy()
                            
                            keypoint_list: List[KeyPoint] = []
                            for j, kp in enumerate(keypoints):
                                # Filter by keypoint confidence threshold
                                if len(kp) >= 3 and kp[2] >= self.keypoint_threshold:
                                    kp_name = self.keypoint_names[j] if j < len(self.keypoint_names) else f"keypoint_{j}"
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
                                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                                keypoints=keypoint_list,
                                confidence=conf,
                                pose_id=i
                            )
                            poses.append(pose)
            elif self.model_type == "mediapipe":
                # Convert BGR to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Run MediaPipe Pose inference
                results = self.model.process(image_rgb)
                
                # Process results
                if results.pose_landmarks:
                    # Extract bounding box from keypoints
                    h, w = image.shape[:2]
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get min and max coordinates for bounding box
                    x_coords = [landmark.x for landmark in landmarks if landmark.visibility > self.keypoint_threshold]
                    y_coords = [landmark.y for landmark in landmarks if landmark.visibility > self.keypoint_threshold]
                    
                    if x_coords and y_coords:
                        x1 = max(0, int(min(x_coords) * w) - 20)
                        y1 = max(0, int(min(y_coords) * h) - 20)
                        x2 = min(w, int(max(x_coords) * w) + 20)
                        y2 = min(h, int(max(y_coords) * h) + 20)
                        
                        # Extract keypoints
                        keypoint_list: List[KeyPoint] = []
                        for j, landmark in enumerate(landmarks):
                            if landmark.visibility > self.keypoint_threshold:
                                # MediaPipe has 33 keypoints, map to COCO format where possible
                                # For simplicity, use MediaPipe keypoint names
                                keypoint_name = f"mediapipe_{j}"
                                keypoint = KeyPoint(
                                    x=float(landmark.x * w),
                                    y=float(landmark.y * h),
                                    confidence=float(landmark.visibility),
                                    keypoint_id=j,
                                    keypoint_name=keypoint_name
                                )
                                keypoint_list.append(keypoint)
                        
                        # Create Pose object
                        pose = Pose(
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            keypoints=keypoint_list,
                            confidence=0.9,  # MediaPipe doesn't provide an overall confidence score
                            pose_id=0  # MediaPipe currently returns one pose per image
                        )
                        poses.append(pose)
            
            return poses
            
        except RuntimeError as e:
            logger.error(f"Runtime error during pose estimation: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error during pose estimation: {e}", exc_info=True)
            return []
    
    def estimate(self, image: np.ndarray) -> List[Pose]:
        """
        Alias for process method
        
        This method is provided for clarity and is functionally equivalent to process().
        
        Args:
            image: Input image in BGR format (numpy array, shape: (H, W, 3))
        
        Returns:
            List[Pose]: List of estimated poses
        """
        return self.process(image)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns information about the currently configured pose estimator, including
        model type, path, device, thresholds, and keypoint configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: "initialized" or "not_initialized"
                - model_type: Type of model ('yolo_pose')
                - model_path: Path to model file
                - device: Device being used ('cpu', 'cuda', 'mps')
                - conf_threshold: Confidence threshold for pose detection
                - keypoint_threshold: Confidence threshold for keypoints
                - num_keypoints: Number of keypoints in the model
            
            If not initialized, returns: {"status": "not_initialized"}
        
        Example:
            ```python
            pose_estimator = PoseEstimator()
            info = pose_estimator.get_model_info()
            print(f"Model: {info.get('model_path')}")
            print(f"Keypoints: {info.get('num_keypoints')}")
            
            pose_estimator.initialize()
            info = pose_estimator.get_model_info()
            print(f"Device: {info.get('device')}")
            ```
        """
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        info: Dict[str, Any] = {
            "status": "initialized",
            "model_type": self.model_type,
            "model_path": self.model_path,
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "keypoint_threshold": self.keypoint_threshold,
            "num_keypoints": len(self.keypoint_names),
        }
        
        return info

