"""
Export utilities for detection and tracking results
"""

import json
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from ..data.detection import Detection
from ..data.track import Track
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ResultExporter:
    """Export detection and tracking results to various formats"""
    
    def __init__(self):
        """Initialize exporter"""
        pass
    
    def export_detections_to_json(
        self,
        detections: List[Detection],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Export detections to JSON file
        
        Args:
            detections: List of Detection objects
            output_path: Output file path
            metadata: Optional metadata to include
            
        Returns:
            bool: True if successful
        """
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "count": len(detections),
                "detections": [det.to_dict() for det in detections]
            }
            
            if metadata:
                data["metadata"] = metadata
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to write JSON file ({output_path}): {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error exporting detections to JSON: {e}", exc_info=True)
            return False
    
    def export_tracks_to_json(
        self,
        tracks: List[Track],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Export tracks to JSON file
        
        Args:
            tracks: List of Track objects
            output_path: Output file path
            metadata: Optional metadata to include
            
        Returns:
            bool: True if successful
        """
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "count": len(tracks),
                "tracks": [track.to_dict() for track in tracks]
            }
            
            if metadata:
                data["metadata"] = metadata
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to write JSON file ({output_path}): {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error exporting tracks to JSON: {e}", exc_info=True)
            return False
    
    def export_detections_to_csv(
        self,
        detections: List[Detection],
        output_path: str
    ) -> bool:
        """
        Export detections to CSV file
        
        Args:
            detections: List of Detection objects
            output_path: Output file path
            
        Returns:
            bool: True if successful
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([
                    "class_id", "class_name", "confidence",
                    "x1", "y1", "x2", "y2", "width", "height"
                ])
                
                # Write data
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    width = x2 - x1
                    height = y2 - y1
                    writer.writerow([
                        det.class_id,
                        det.class_name or "",
                        f"{det.confidence:.4f}",
                        f"{x1:.2f}", f"{y1:.2f}",
                        f"{x2:.2f}", f"{y2:.2f}",
                        f"{width:.2f}", f"{height:.2f}"
                    ])
            
            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to write CSV file ({output_path}): {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error exporting detections to CSV: {e}", exc_info=True)
            return False
    
    def export_tracks_to_csv(
        self,
        tracks: List[Track],
        output_path: str
    ) -> bool:
        """
        Export tracks to CSV file
        
        Args:
            tracks: List of Track objects
            output_path: Output file path
            
        Returns:
            bool: True if successful
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([
                    "track_id", "class_id", "class_name", "confidence",
                    "x1", "y1", "x2", "y2", "width", "height",
                    "age", "time_since_update"
                ])
                
                # Write data
                for track in tracks:
                    x1, y1, x2, y2 = track.bbox
                    width = x2 - x1
                    height = y2 - y1
                    writer.writerow([
                        track.track_id,
                        track.class_id,
                        track.class_name or "",
                        f"{track.confidence:.4f}",
                        f"{x1:.2f}", f"{y1:.2f}",
                        f"{x2:.2f}", f"{y2:.2f}",
                        f"{width:.2f}", f"{height:.2f}",
                        track.age,
                        track.time_since_update
                    ])
            
            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to write CSV file ({output_path}): {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error exporting tracks to CSV: {e}", exc_info=True)
            return False
    
    def export_video_results_to_json(
        self,
        video_results: List[Dict[str, Any]],
        output_path: str,
        video_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Export video processing results to JSON
        
        Args:
            video_results: List of frame results, each containing detections/tracks
            output_path: Output file path
            video_info: Optional video information
            
        Returns:
            bool: True if successful
        """
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "total_frames": len(video_results),
                "frames": video_results
            }
            
            if video_info:
                data["video_info"] = video_info
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to write JSON file ({output_path}): {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error exporting video results to JSON: {e}", exc_info=True)
            return False
    
    def export_to_coco_format(
        self,
        detections: List[Detection],
        image_id: int,
        image_info: Dict[str, Any],
        output_path: str
    ) -> bool:
        """
        Export detections to COCO format
        
        Args:
            detections: List of Detection objects
            image_id: Image ID
            image_info: Image information (width, height, file_name)
            output_path: Output file path
            
        Returns:
            bool: True if successful
        """
        try:
            coco_data = {
                "images": [{
                    "id": image_id,
                    "width": image_info.get("width", 0),
                    "height": image_info.get("height", 0),
                    "file_name": image_info.get("file_name", "")
                }],
                "annotations": [],
                "categories": []
            }
            
            # Get unique categories
            categories = {}
            for det in detections:
                if det.class_id not in categories:
                    categories[det.class_id] = {
                        "id": det.class_id,
                        "name": det.class_name or f"class_{det.class_id}",
                        "supercategory": "none"
                    }
            
            coco_data["categories"] = list(categories.values())
            
            # Add annotations
            for idx, det in enumerate(detections):
                x1, y1, x2, y2 = det.bbox
                width = x2 - x1
                height = y2 - y1
                
                annotation = {
                    "id": idx,
                    "image_id": image_id,
                    "category_id": det.class_id,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "score": det.confidence
                }
                coco_data["annotations"].append(annotation)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, indent=2, ensure_ascii=False)
            
            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to write COCO format file ({output_path}): {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error exporting to COCO format: {e}", exc_info=True)
            return False

