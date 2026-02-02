#!/usr/bin/env python3
"""
Example demonstrating SAM (Segment Anything Model) segmentation features

This example shows how to use the SAM segmenter for:
1. Automatic segmentation
2. Interactive segmentation with points
3. Detection + segmentation combined inference
4. Batch processing with segmentation

Usage:
    python 08_segmentation_sam.py [--input input_image.jpg]
    python 08_segmentation_sam.py --camera 0
"""

import argparse
import cv2
import numpy as np
from typing import List
from visionframework import Detector, SAMSegmenter, VisionFrameworkError, Visualizer, Detection


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Vision Framework SAM Segmentation Example')
    parser.add_argument('--input', type=str, default='', help='Input image file path')
    parser.add_argument('--camera', type=int, default=-1, help='Camera index (default: -1 for image file)')
    parser.add_argument('--output', type=str, help='Output image file path')
    parser.add_argument('--show', action='store_true', default=True, help='Show output')
    parser.add_argument('--model_type', type=str, default='vit_b', 
                       choices=['vit_h', 'vit_l', 'vit_b'], help='SAM model type')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    return parser.parse_args()


def draw_masks(frame: np.ndarray, masks: List[dict], alpha: float = 0.5) -> np.ndarray:
    """Draw segmentation masks on the frame"""
    # Convert masks to Detection objects
    detections = []
    for i, mask in enumerate(masks):
        if 'segmentation' in mask:
            x, y, w, h = mask.get('bbox', [0, 0, 0, 0])
            detection = Detection(
                class_id=i,
                class_name=f"Mask {i+1}",
                confidence=1.0,
                x=x,
                y=y,
                width=w,
                height=h,
                mask=mask['segmentation']
            )
            detections.append(detection)
    
    # Use the integrated Visualizer to draw detections with masks
    viz = Visualizer()
    return viz.draw_detections(frame, detections)


def draw_detections_with_masks(frame: np.ndarray, detections: List, alpha: float = 0.5) -> np.ndarray:
    """Draw detections with segmentation masks"""
    viz = Visualizer()
    # Use the integrated Visualizer to draw detections with masks
    return viz.draw_detections(frame, detections)


def automatic_segmentation_example(image: np.ndarray, args):
    """Example of automatic segmentation using SAM"""
    print("=== Automatic Segmentation Example ===")
    
    # Initialize SAM segmenter
    sam = SAMSegmenter({
        "model_type": args.model_type,
        "device": args.device,
        "use_fp16": args.device != "cpu",
        "automatic_threshold": 0.8
    })
    
    if not sam.initialize():
        print("Failed to initialize SAM segmenter")
        return image.copy()
    
    # Perform automatic segmentation
    masks = sam.automatic_segment(image)
    print(f"Found {len(masks)} masks")
    
    # Draw masks on the image
    result = draw_masks(image, masks)
    
    # Cleanup
    sam.cleanup()
    
    return result


def detection_plus_segmentation_example(image: np.ndarray, args):
    """Example of combined detection and segmentation"""
    print("=== Detection + Segmentation Example ===")
    
    # Initialize detector with SAM segmenter
    detector = Detector({
        "model_path": "yolov8n.pt",
        "model_type": "yolo",
        "conf_threshold": 0.3,
        "device": args.device,
        "segmenter_type": "sam",
        "sam_model_type": args.model_type,
        "sam_use_fp16": args.device != "cpu"
    })
    
    if not detector.initialize():
        print("Failed to initialize detector with SAM segmenter")
        return image.copy()
    
    # Perform detection + segmentation
    detections = detector.detect(image)
    print(f"Detected {len(detections)} objects, {sum(1 for d in detections if hasattr(d, 'mask') and d.mask is not None)} with masks")
    
    # Draw detections with masks
    result = draw_detections_with_masks(image, detections)
    
    # Cleanup
    detector.cleanup()
    
    return result


def interactive_segmentation_example(image: np.ndarray, args):
    """Example of interactive segmentation with points"""
    print("=== Interactive Segmentation Example ===")
    print("Click on the object you want to segment, then press 'q' to continue")
    
    # Initialize SAM segmenter
    sam = SAMSegmenter({
        "model_type": args.model_type,
        "device": args.device,
        "use_fp16": args.device != "cpu"
    })
    
    if not sam.initialize():
        print("Failed to initialize SAM segmenter")
        return image.copy()
    
    # Store click points
    points = []
    labels = []
    result = image.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, labels, result
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            labels.append(1)  # Positive point
            
            # Draw point
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(result, f"+{len(points)}", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Update segmentation
            if len(points) >= 1:
                masks = sam.segment_with_points(image, points, labels)
                if masks:
                    # Draw the best mask
                    best_mask = masks[0]
                    colored_mask = np.zeros_like(image)
                    colored_mask[best_mask['segmentation'] > 0.5] = (0, 255, 0)
                    result = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
                    
                    # Redraw points on top
                    for i, (px, py) in enumerate(points):
                        cv2.circle(result, (px, py), 5, (0, 255, 0), -1)
                        cv2.putText(result, f"+{i+1}", (px+10, py-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((x, y))
            labels.append(0)  # Negative point
            
            # Draw point
            cv2.circle(result, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(result, f"-{len(points)}", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Show interactive window
    cv2.namedWindow('Interactive Segmentation', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Interactive Segmentation', mouse_callback)
    
    while True:
        cv2.imshow('Interactive Segmentation', result)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyWindow('Interactive Segmentation')
    
    # Cleanup
    sam.cleanup()
    
    return result


def main():
    """Main function"""
    args = parse_args()
    
    # Load or capture image
    if args.camera >= 0:
        print(f"Using camera {args.camera}")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Error: Could not open camera {args.camera}")
            return
        ret, image = cap.read()
        cap.release()
        if not ret:
            print("Error: Could not capture image from camera")
            return
    elif args.input:
        print(f"Loading image from {args.input}")
        image = cv2.imread(args.input)
        if image is None:
            print(f"Error: Could not read image from {args.input}")
            return
    else:
        print("Error: Please specify either --input or --camera")
        return
    
    # Resize for better display
    max_size = 800
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    
    try:
        # Run examples
        examples = [
            ("Automatic Segmentation", automatic_segmentation_example),
            ("Detection + Segmentation", detection_plus_segmentation_example),
            ("Interactive Segmentation", interactive_segmentation_example)
        ]
        
        results = []
        for name, func in examples:
            print(f"\n--- {name} ---")
            result = func(image.copy(), args)
            results.append((name, result))
        
        # Show results
        if args.show:
            for name, result in results:
                window_name = f"Result: {name}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, result)
            
            print("\nPress any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Save results if specified
        if args.output:
            for i, (name, result) in enumerate(results):
                output_path = args.output
                if len(results) > 1:
                    # Add example index to filename
                    import os
                    base, ext = os.path.splitext(args.output)
                    output_path = f"{base}_{i+1}{ext}"
                
                cv2.imwrite(output_path, result)
                print(f"Saved result to {output_path}")
        
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
