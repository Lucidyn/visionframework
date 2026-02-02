#!/usr/bin/env python3
"""
Example demonstrating CLIP (Contrastive Language-Image Pre-training) features

This example shows how to use the CLIP extractor for:
1. Image-text similarity calculation
2. Zero-shot classification
3. Image feature extraction
4. Different CLIP model variants

Usage:
    python 09_clip_features.py [--input input_image.jpg]
    python 09_clip_features.py --camera 0
"""

import argparse
import cv2
import numpy as np
from typing import List, Tuple
from visionframework import CLIPExtractor, VisionFrameworkError


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Vision Framework CLIP Features Example')
    parser.add_argument('--input', type=str, default='', help='Input image file path')
    parser.add_argument('--camera', type=int, default=-1, help='Camera index (default: -1 for image file)')
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-base-patch32',
                       help='CLIP model name to use')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--show', action='store_true', default=True, help='Show output')
    return parser.parse_args()


def draw_clip_results(frame: np.ndarray, text: str, similarity: float, classification: str) -> np.ndarray:
    """Draw CLIP results on the frame"""
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Create a black background for text
    text_bg = np.zeros((120, w, 3), dtype=np.uint8)
    
    # Draw results
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Text similarity
    sim_text = f"Text similarity: {text} ({similarity:.2f})"
    cv2.putText(text_bg, sim_text, (10, 30), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    
    # Zero-shot classification
    cls_text = f"Zero-shot classification: {classification}"
    cv2.putText(text_bg, cls_text, (10, 60), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    
    # Combine with original frame
    result = np.vstack((frame, text_bg))
    
    return result


def text_image_similarity_example(image: np.ndarray, clip: CLIPExtractor) -> Tuple[str, float]:
    """Example of image-text similarity calculation"""
    print("=== Image-Text Similarity Example ===")
    
    # Define test texts
    test_texts = [
        "a cat",
        "a dog", 
        "a person",
        "a car",
        "a tree",
        "a building"
    ]
    
    # Calculate similarities
    similarities = clip.image_text_similarity(image, test_texts)
    
    # Find the best match
    best_idx = np.argmax(similarities)
    best_text = test_texts[best_idx]
    best_similarity = similarities[best_idx]
    
    print(f"Image is most similar to: '{best_text}' with similarity: {best_similarity:.2f}")
    print("All similarities:")
    for text, sim in zip(test_texts, similarities):
        print(f"  '{text}': {sim:.2f}")
    
    return best_text, best_similarity


def zero_shot_classification_example(image: np.ndarray, clip: CLIPExtractor) -> str:
    """Example of zero-shot classification"""
    print("\n=== Zero-shot Classification Example ===")
    
    # Define classification categories
    categories = [
        "cat",
        "dog",
        "person",
        "vehicle",
        "animal",
        "object",
        "scene"
    ]
    
    # Perform zero-shot classification
    results = clip.zero_shot_classification(image, categories)
    
    # Find the best category
    best_category = results["best_category"]
    best_score = results["best_score"]
    
    print(f"Zero-shot classification result: {best_category} ({best_score:.2f})")
    print("All classification scores:")
    for category, score in zip(results["categories"], results["scores"]):
        print(f"  {category}: {score:.2f}")
    
    return f"{best_category} ({best_score:.2f})"


def image_feature_extraction_example(image: np.ndarray, clip: CLIPExtractor):
    """Example of image feature extraction"""
    print("\n=== Image Feature Extraction Example ===")
    
    # Extract image features
    features = clip.extract_image_features(image)
    
    print(f"Image features shape: {features.shape}")
    print(f"Features dtype: {features.dtype}")
    print(f"Features mean: {features.mean():.4f}, std: {features.std():.4f}")
    
    return features


def main():
    """Main function"""
    args = parse_args()
    
    try:
        # Initialize CLIP extractor
        clip = CLIPExtractor({
            "model_name": args.model_name,
            "device": args.device,
            "use_fp16": args.device != "cpu"
        })
        
        if not clip.initialize():
            print("Failed to initialize CLIP extractor")
            return
        
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
            # Create a simple test image with different objects
            print("Creating test image...")
            image = np.zeros((400, 600, 3), dtype=np.uint8)
            
            # Draw a red rectangle (like a car)
            cv2.rectangle(image, (50, 200), (150, 300), (0, 0, 255), -1)
            
            # Draw a green circle (like a ball)
            cv2.circle(image, (300, 250), 50, (0, 255, 0), -1)
            
            # Draw a blue triangle (like a roof)
            points = np.array([[450, 200], [500, 300], [550, 200]], np.int32)
            cv2.fillPoly(image, [points], (255, 0, 0))
        
        # Resize for better display
        max_size = 600
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        
        print(f"Using CLIP model: {args.model_name}")
        print(f"Device: {clip.device}")
        
        # Run examples
        best_text, best_similarity = text_image_similarity_example(image, clip)
        classification = zero_shot_classification_example(image, clip)
        _ = image_feature_extraction_example(image, clip)
        
        # Draw results
        result = draw_clip_results(image, best_text, best_similarity, classification)
        
        # Show results
        if args.show:
            cv2.namedWindow('CLIP Features Example', cv2.WINDOW_NORMAL)
            cv2.imshow('CLIP Features Example', result)
            print("\nPress any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Cleanup
        clip.cleanup()
        
    except ImportError as e:
        print(f"Error: Missing dependencies for CLIP. Please install them with: pip install transformers torch numpy")
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