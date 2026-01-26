#!/usr/bin/env python3
"""
Example demonstrating pose estimation features

This example shows how to use the pose estimator for:
1. YOLO Pose and MediaPipe Pose models
2. Keypoint detection and visualization
3. Different model types and configurations
4. Confidence threshold filtering

Usage:
    python 10_pose_estimation.py [--input input_image.jpg]
    python 10_pose_estimation.py --camera 0
    python 10_pose_estimation.py --model_type mediapipe
"""

import argparse
import cv2
import numpy as np
from typing import List, Dict, Any
from visionframework.core.pose_estimator import PoseEstimator
from visionframework.exceptions import VisionFrameworkError


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Vision Framework Pose Estimation Example')
    parser.add_argument('--input', type=str, default='', help='Input image file path')
    parser.add_argument('--camera', type=int, default=-1, help='Camera index (default: -1 for image file)')
    parser.add_argument('--model_type', type=str, default='yolo',
                       choices=['yolo', 'mediapipe'], help='Pose estimation model type')
    parser.add_argument('--model_path', type=str, default='yolov8n-pose.pt',
                       help='YOLO pose model path')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                       help='Confidence threshold for pose detection')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--show', action='store_true', default=True, help='Show output')
    return parser.parse_args()


def draw_poses(frame: np.ndarray, poses: List[Dict[str, Any]], model_type: str) -> np.ndarray:
    """Draw pose keypoints and connections on the frame"""
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Define keypoint colors based on body parts
    keypoint_colors = {
        # Face
        'nose': (0, 255, 0),
        'left_eye': (0, 255, 0),
        'right_eye': (0, 255, 0),
        'left_ear': (0, 255, 0),
        'right_ear': (0, 255, 0),
        
        # Torso
        'left_shoulder': (255, 0, 0),
        'right_shoulder': (255, 0, 0),
        'left_elbow': (255, 0, 0),
        'right_elbow': (255, 0, 0),
        'left_wrist': (255, 0, 0),
        'right_wrist': (255, 0, 0),
        
        # Lower body
        'left_hip': (0, 0, 255),
        'right_hip': (0, 0, 255),
        'left_knee': (0, 0, 255),
        'right_knee': (0, 0, 255),
        'left_ankle': (0, 0, 255),
        'right_ankle': (0, 0, 255),
    }
    
    # Define connections between keypoints
    connections = [
        # Face
        ('left_eye', 'nose'),
        ('right_eye', 'nose'),
        ('left_eye', 'left_ear'),
        ('right_eye', 'right_ear'),
        
        # Torso
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('right_shoulder', 'right_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_elbow', 'right_wrist'),
        
        # Body
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        
        # Legs
        ('left_hip', 'left_knee'),
        ('right_hip', 'right_knee'),
        ('left_knee', 'left_ankle'),
        ('right_knee', 'right_ankle'),
    ]
    
    # Draw each pose
    for i, pose in enumerate(poses):
        keypoints = pose.get('keypoints', {})
        confidence = pose.get('confidence', 0.0)
        
        # Draw bounding box if available
        if 'bbox' in pose:
            x, y, width, height = pose['bbox']
            cv2.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)),
                        (0, 255, 0), 2, cv2.LINE_AA)
            
            # Draw confidence score
            cv2.putText(frame, f"Pose {i+1}: {confidence:.2f}", 
                       (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Draw keypoints
        for kp_name, kp in keypoints.items():
            x, y, conf = kp
            if conf >= 0.5:  # Only draw keypoints with high confidence
                color = keypoint_colors.get(kp_name, (255, 255, 255))
                cv2.circle(frame, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
                
                # Draw keypoint name
                cv2.putText(frame, kp_name[:3], (int(x) + 10, int(y) + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
        
        # Draw connections
        for conn_start, conn_end in connections:
            if conn_start in keypoints and conn_end in keypoints:
                start_x, start_y, start_conf = keypoints[conn_start]
                end_x, end_y, end_conf = keypoints[conn_end]
                
                if start_conf >= 0.5 and end_conf >= 0.5:
                    color = (255, 255, 0)  # Yellow connections
                    cv2.line(frame, (int(start_x), int(start_y)),
                            (int(end_x), int(end_y)), color, 2, cv2.LINE_AA)
    
    return frame


def draw_model_info(frame: np.ndarray, model_type: str, device: str) -> np.ndarray:
    """Draw model information on the frame"""
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Create info text
    info_text = [
        f"Pose Estimation - Model: {model_type}",
        f"Device: {device}",
        "Press 'q' to quit",
        "Press 'm' to toggle model type"
    ]
    
    # Draw info on top-left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    y = 20
    
    for text in info_text:
        cv2.putText(frame, text, (10, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += 20
    
    return frame


def create_test_person_image() -> np.ndarray:
    """Create a simple test image with a stick figure representing a person"""
    print("Creating test person image...")
    
    # Create a white background
    image = np.ones((400, 300, 3), dtype=np.uint8) * 255
    
    # Draw a simple stick figure
    center_x, center_y = 150, 150
    
    # Head
    cv2.circle(image, (center_x, center_y - 60), 20, (0, 0, 0), -1, cv2.LINE_AA)
    
    # Body
    cv2.line(image, (center_x, center_y - 40), (center_x, center_y + 60), (0, 0, 0), 5, cv2.LINE_AA)
    
    # Arms
    cv2.line(image, (center_x, center_y - 20), (center_x - 40, center_y - 50), (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(image, (center_x, center_y - 20), (center_x + 40, center_y - 50), (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(image, (center_x - 40, center_y - 50), (center_x - 50, center_y - 70), (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(image, (center_x + 40, center_y - 50), (center_x + 50, center_y - 70), (0, 0, 0), 5, cv2.LINE_AA)
    
    # Legs
    cv2.line(image, (center_x, center_y + 60), (center_x - 40, center_y + 120), (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(image, (center_x, center_y + 60), (center_x + 40, center_y + 120), (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(image, (center_x - 40, center_y + 120), (center_x - 50, center_y + 140), (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(image, (center_x + 40, center_y + 120), (center_x + 50, center_y + 140), (0, 0, 0), 5, cv2.LINE_AA)
    
    return image


def test_different_model_types(args):
    """Test different pose estimation model types"""
    print("=== Testing Different Model Types ===")
    
    model_types = ['yolo', 'mediapipe']
    
    for model_type in model_types:
        print(f"\nTesting {model_type} model...")
        
        # Initialize pose estimator with current model type
        pose_estimator = PoseEstimator({
            "model_type": model_type,
            "model_path": args.model_path,
            "conf_threshold": args.conf_threshold,
            "device": args.device
        })
        
        if pose_estimator.initialize():
            print(f"✓ {model_type} model initialized successfully")
            
            # Create test image
            test_image = create_test_person_image()
            
            # Run pose estimation
            poses = pose_estimator.estimate(test_image)
            print(f"✓ Detected {len(poses)} poses with {model_type} model")
            
            if poses:
                # Print pose details
                first_pose = poses[0]
                print(f"  First pose confidence: {first_pose.get('confidence', 0.0):.2f}")
                print(f"  Keypoints detected: {len(first_pose.get('keypoints', {}))}")
            
            # Cleanup
            pose_estimator.cleanup()
        else:
            print(f"✗ Failed to initialize {model_type} model")


def main():
    """Main function"""
    args = parse_args()
    
    # Test different model types first
    test_different_model_types(args)
    
    try:
        # Initialize pose estimator
        pose_estimator = PoseEstimator({
            "model_type": args.model_type,
            "model_path": args.model_path,
            "conf_threshold": args.conf_threshold,
            "device": args.device
        })
        
        if not pose_estimator.initialize():
            print(f"Failed to initialize {args.model_type} pose estimator")
            return
        
        # Load or capture image
        if args.camera >= 0:
            print(f"Using camera {args.camera}")
            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                print(f"Error: Could not open camera {args.camera}")
                return
            
            print("Press 'q' to quit, 'm' to toggle model type")
            current_model = args.model_type
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Estimate poses
                poses = pose_estimator.estimate(frame)
                
                # Draw results
                result = draw_poses(frame, poses, current_model)
                result = draw_model_info(result, current_model, pose_estimator.device)
                
                # Show output
                if args.show:
                    cv2.imshow('Pose Estimation Example', result)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('m'):
                        # Toggle model type
                        current_model = 'mediapipe' if current_model == 'yolo' else 'yolo'
                        print(f"Switching to {current_model} model")
                        
                        # Reinitialize pose estimator with new model type
                        pose_estimator.cleanup()
                        pose_estimator = PoseEstimator({
                            "model_type": current_model,
                            "model_path": args.model_path,
                            "conf_threshold": args.conf_threshold,
                            "device": args.device
                        })
                        pose_estimator.initialize()
            
            cap.release()
            if args.show:
                cv2.destroyAllWindows()
                
        elif args.input:
            print(f"Loading image from {args.input}")
            image = cv2.imread(args.input)
            if image is None:
                print(f"Error: Could not read image from {args.input}")
                return
        else:
            # Create a test image
            image = create_test_person_image()
        
        if args.input or args.camera < 0:  # Process single image
            print(f"\nRunning pose estimation with {args.model_type} model...")
            
            # Estimate poses
            poses = pose_estimator.estimate(image)
            print(f"Detected {len(poses)} poses")
            
            # Draw results
            result = draw_poses(image, poses, args.model_type)
            result = draw_model_info(result, args.model_type, pose_estimator.device)
            
            # Show results
            if args.show:
                cv2.namedWindow('Pose Estimation Example', cv2.WINDOW_NORMAL)
                cv2.imshow('Pose Estimation Example', result)
                print("Press any key to exit...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        # Cleanup
        pose_estimator.cleanup()
        
    except ImportError as e:
        print(f"Error: Missing dependencies for pose estimation. Please install required packages.")
        print(f"Detailed error: {e}")
    except VisionFrameworkError as e:
        print(f"Pipeline error: {e}")
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    except Exception as e:
        import traceback
        print(f"Unexpected error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()