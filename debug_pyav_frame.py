"""
Debug script to check PyAV frame format
"""

import cv2
import numpy as np
from visionframework.utils.io import PyAVVideoProcessor

# Try to import pyav
try:
    import av
    pyav_available = True
except ImportError:
    pyav_available = False

print("=== Debug PyAV Frame Format ===")
print(f"PyAV available: {pyav_available}")

# Create a simple test video
def create_test_video():
    """Create a simple test video"""
    test_video_path = "debug_test.mp4"
    height, width = 240, 320
    fps = 15
    total_frames = 5
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(test_video_path, fourcc, fps, (width, height))
    
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)
    
    writer.release()
    return test_video_path

# Debug PyAV frame format
def debug_pyav_frame():
    """Debug PyAV frame format"""
    print("\nDebugging PyAV frame format...")
    
    # Create test video
    test_video = create_test_video()
    print(f"Created test video: {test_video}")
    
    try:
        # Test PyAVVideoProcessor
        processor = PyAVVideoProcessor(test_video)
        print("✓ PyAVVideoProcessor initialized")
        
        if processor.open():
            print("✓ Video opened successfully")
            
            # Read a frame
            ret, frame = processor.read_frame()
            if ret:
                print(f"✓ Read frame successfully")
                print(f"  Frame shape: {frame.shape}")
                print(f"  Frame dtype: {frame.dtype}")
                print(f"  Frame min: {frame.min()}")
                print(f"  Frame max: {frame.max()}")
                print(f"  Frame flags: {frame.flags}")
                
                # Test OpenCV operations
                print("\nTesting OpenCV operations:")
                
                # Test 1: Convert to grayscale
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    print("  ✓ cv2.cvtColor works")
                except Exception as e:
                    print(f"  ✗ cv2.cvtColor failed: {e}")
                
                # Test 2: Put text
                try:
                    # Make a copy to avoid modifying original
                    frame_copy = frame.copy()
                    cv2.putText(frame_copy, "Test", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    print("  ✓ cv2.putText works")
                except Exception as e:
                    print(f"  ✗ cv2.putText failed: {e}")
                
                # Test 3: Check if frame is contiguous
                print(f"  Frame is contiguous: {frame.flags.contiguous}")
                
                # Test 4: Try making it contiguous if not
                if not frame.flags.contiguous:
                    print("  Making frame contiguous...")
                    frame_contiguous = np.ascontiguousarray(frame)
                    print(f"  New frame is contiguous: {frame_contiguous.flags.contiguous}")
                    
                    try:
                        cv2.putText(frame_contiguous, "Test", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        print("  ✓ cv2.putText works on contiguous frame")
                    except Exception as e:
                        print(f"  ✗ cv2.putText still failed: {e}")
            else:
                print("✗ Failed to read frame")
            
            processor.close()
        else:
            print("✗ Failed to open video")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import os
        if os.path.exists(test_video):
            os.remove(test_video)
            print(f"Cleaned up: {test_video}")

if __name__ == "__main__":
    if pyav_available:
        debug_pyav_frame()
    else:
        print("PyAV not available, skipping debug")
