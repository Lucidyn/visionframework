#!/usr/bin/env python3
"""
Example demonstrating enhanced features of the Vision Framework

This example shows how to use the enhanced features including:
1. Enhanced ReID tracking with multiple model support
2. Trajectory analysis and prediction
3. Performance monitoring with GPU support
4. Batch processing optimization
5. FP16 precision for improved performance

Usage:
    python 07_enhanced_features.py [--input input_video.mp4]
    python 07_enhanced_features.py --camera 0
"""

import argparse
import cv2
import numpy as np
from typing import List
from visionframework import VisionPipeline, Track, Visualizer, VisionFrameworkError
from visionframework.utils import PerformanceMonitor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Vision Framework Enhanced Features Example')
    parser.add_argument('--input', type=str, help='Input video file path')
    parser.add_argument('--camera', type=int, default=-1, help='Camera index (default: -1 for video file)')
    parser.add_argument('--output', type=str, help='Output video file path')
    parser.add_argument('--show', action='store_true', default=True, help='Show output video')
    return parser.parse_args()


def setup_pipeline():
    """Setup pipeline with enhanced features"""
    # Create pipeline with enhanced configuration
    pipeline_config = {
        "detector_config": {
            "model_path": "yolov8n.pt",
            "model_type": "yolo",
            "conf_threshold": 0.3,
            "device": "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
            "use_fp16": True,  # Enable FP16 precision for faster inference
            "batch_inference": True,
            "dynamic_batch_size": True,
            "max_batch_size": 8
        },
        "tracker_config": {
            "tracker_type": "reid",  # Use ReID tracker with enhanced features
            "max_age": 30,
            "min_hits": 3,
            "iou_threshold": 0.3,
            "reid_weight": 0.7,
            "trajectory_analysis": True,  # Enable trajectory analysis
            "fps": 30.0
        },
        "enable_tracking": True
    }
    
    # Initialize pipeline
    pipeline = VisionPipeline(pipeline_config)
    pipeline.initialize()
    
    return pipeline


def setup_performance_monitor():
    """Setup enhanced performance monitor"""
    return PerformanceMonitor(
        window_size=30,
        enabled_metrics=["fps", "component_times", "memory", "gpu"],
        enable_gpu_monitoring=True  # Enable GPU monitoring
    )


def draw_analysis_results(frame: np.ndarray, tracks: List[Track], pipeline: VisionPipeline):
    """Draw trajectory analysis results on frame"""
    # Use the integrated Visualizer to draw tracks
    viz = Visualizer()
    frame = viz.draw_tracks(frame, tracks, draw_history=True)
    
    # Add additional analysis information
    for track in tracks:
        if len(track.history) < 2:
            continue
        
        # Get track analysis
        analysis = pipeline.tracker.analyze_track(track)
        if not analysis:
            continue
        
        # Draw speed information
        x1, y1, x2, y2 = track.bbox
        speed_mag = analysis["speed"]["magnitude"]
        speed_text = f"Speed: {speed_mag:.2f} px/frame"
        cv2.putText(frame, speed_text, (int(x1), int(y1) - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw direction arrow
        direction = analysis["direction"]
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Calculate arrow end point based on direction
        arrow_length = 30
        end_x = int(center_x + arrow_length * np.cos(np.radians(direction)))
        end_y = int(center_y + arrow_length * np.sin(np.radians(direction)))
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                       (255, 0, 0), 2, tipLength=0.3)
        
        # Predict next position
        predicted_bbox = pipeline.tracker.predict_next_position(track, frames_ahead=2)
        px1, py1, px2, py2 = predicted_bbox
        cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), 
                    (0, 165, 255), 2, cv2.LINE_AA)  # Orange dashed box


def draw_performance_metrics(frame: np.ndarray, monitor: PerformanceMonitor):
    """Draw performance metrics on frame"""
    metrics = monitor.get_metrics()
    
    # Prepare metrics text
    metrics_text = [
        f"FPS: {metrics.fps:.1f} (Avg: {metrics.avg_fps:.1f})",
        f"Frame Time: {metrics.avg_time_per_frame*1000:.1f} ms",
        f"Memory: {metrics.avg_memory_usage:.1f} MB",
    ]
    
    # Add GPU metrics if available
    if metrics.gpu_memory_usage > 0:
        metrics_text.append(f"GPU Mem: {metrics.gpu_memory_usage:.1f} MB")
        metrics_text.append(f"GPU Util: {metrics.gpu_utilization:.1f}%")
    
    # Draw component times
    metrics_text.append("Component Times (ms):")
    for component, time_ms in metrics.component_times.items():
        if time_ms > 0:
            metrics_text.append(f"  {component}: {time_ms:.1f}")
    
    # Draw text on frame
    y = 20
    for text in metrics_text:
        cv2.putText(frame, text, (10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 15


def process_video(args):
    """Process video with enhanced features"""
    # Setup pipeline
    pipeline = setup_pipeline()
    
    # Setup performance monitor
    perf_monitor = setup_performance_monitor()
    perf_monitor.start()
    
    # Setup video capture
    if args.camera >= 0:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Error: Could not open camera {args.camera}")
            return
    else:
        if not args.input:
            print("Error: Either --input or --camera must be specified")
            return
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Error: Could not open video file {args.input}")
            return
    
    # Setup video writer if needed
    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_count = 0
    tracks_history = {}
    
    try:
        print("Processing started. Press 'q' to quit.")
        print("Enhanced features enabled:")
        print("  - ReID tracking with ResNet50")
        print("  - Trajectory analysis and prediction")
        print("  - Performance monitoring with GPU support")
        print("  - FP16 precision for faster inference")
        print("  - Dynamic batch processing")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Start performance tick
            perf_monitor.tick()
            
            # Process frame with pipeline
            results = pipeline.process(frame)
            tracks = results.tracks
            
            # Update performance monitor with component times
            if hasattr(results, 'component_times'):
                for component, time_ms in results.component_times.items():
                    perf_monitor.record_component_time(component, time_ms / 1000)  # Convert to seconds
            
            # Store track history for trajectory analysis
            for track in tracks:
                if track.track_id not in tracks_history:
                    tracks_history[track.track_id] = track
                else:
                    tracks_history[track.track_id] = track
            
            # Draw analysis results
            draw_analysis_results(frame, tracks, pipeline)
            
            # Draw performance metrics
            draw_performance_metrics(frame, perf_monitor)
            
            # Show output
            if args.show:
                cv2.imshow('Vision Framework - Enhanced Features', frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write to output file if specified
            if writer:
                writer.write(frame)
            
            # Print analysis summary every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}:")
                print(f"  FPS: {perf_monitor.get_metrics().fps:.1f}")
                print(f"  Active tracks: {len(tracks)}")
                
                # Analyze all tracks
                if tracks:
                    analysis_results = pipeline.tracker.analyze_tracks(tracks)
                    if analysis_results:
                        avg_speed = np.mean([r['speed']['magnitude'] for r in analysis_results])
                        print(f"  Avg track speed: {avg_speed:.2f} pixels/frame")
            
    except VisionFrameworkError as e:
        print(f"Pipeline error: {e}")
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        
        pipeline.cleanup()
        
        # Print final performance report
        print("\n=== Final Performance Report ===")
        report = perf_monitor.get_detailed_report()
        print(f"Average FPS: {report['fps']['average']:.1f}")
        print(f"Average Frame Time: {report['frame_times']['average']*1000:.1f} ms")
        print(f"Peak Memory: {report['memory']['peak_mb']:.1f} MB")
        
        if 'gpu' in report:
            print(f"Average GPU Memory: {report['gpu']['memory_usage_mb']:.1f} MB")
            print(f"Average GPU Utilization: {report['gpu']['utilization_percent']:.1f}%")
        
        print("Component Times (ms):")
        for component, time_ms in report['component_times_ms'].items():
            if time_ms > 0:
                print(f"  {component}: {time_ms:.1f}")


if __name__ == "__main__":
    args = parse_args()
    process_video(args)
