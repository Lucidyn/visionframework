"""
Trajectory analysis utilities
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ..data.track import Track


class TrajectoryAnalyzer:
    """Analyze object trajectories"""
    
    def __init__(self, fps: float = 30.0, pixel_to_meter: Optional[float] = None):
        """
        Initialize trajectory analyzer
        
        Args:
            fps: Frames per second for speed calculation
            pixel_to_meter: Conversion factor from pixels to meters (for real-world speed)
        """
        self.fps = fps
        self.pixel_to_meter = pixel_to_meter
    
    def calculate_speed(
        self,
        track: Track,
        use_real_world: bool = False
    ) -> Tuple[float, float]:
        """
        Calculate speed (pixels/frame or m/s)
        
        Args:
            track: Track object
            use_real_world: If True, convert to m/s using pixel_to_meter
            
        Returns:
            Tuple of (speed_x, speed_y) in pixels/frame or m/s
        """
        if len(track.history) < 2:
            return (0.0, 0.0)
        
        # Get last two positions
        prev_bbox = track.history[-2]
        curr_bbox = track.history[-1]
        
        # Calculate center positions
        prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
        prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
        curr_center_x = (curr_bbox[0] + curr_bbox[2]) / 2
        curr_center_y = (curr_bbox[1] + curr_bbox[3]) / 2
        
        # Calculate speed
        speed_x = curr_center_x - prev_center_x
        speed_y = curr_center_y - prev_center_y
        
        if use_real_world and self.pixel_to_meter:
            speed_x *= self.pixel_to_meter * self.fps
            speed_y *= self.pixel_to_meter * self.fps
        
        return (speed_x, speed_y)
    
    def calculate_direction(
        self,
        track: Track,
        use_degrees: bool = True
    ) -> float:
        """
        Calculate movement direction
        
        Args:
            track: Track object
            use_degrees: If True, return degrees; else radians
            
        Returns:
            Direction angle (0-360 degrees or 0-2π radians)
        """
        if len(track.history) < 2:
            return 0.0
        
        speed_x, speed_y = self.calculate_speed(track)
        
        # Calculate angle
        angle = np.arctan2(speed_y, speed_x)
        
        if use_degrees:
            angle = np.degrees(angle)
            if angle < 0:
                angle += 360
        
        return angle
    
    def calculate_total_distance(self, track: Track) -> float:
        """Calculate total distance traveled"""
        if len(track.history) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(track.history)):
            prev_bbox = track.history[i-1]
            curr_bbox = track.history[i]
            
            prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
            prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
            curr_center_x = (curr_bbox[0] + curr_bbox[2]) / 2
            curr_center_y = (curr_bbox[1] + curr_bbox[3]) / 2
            
            distance = np.sqrt(
                (curr_center_x - prev_center_x)**2 + 
                (curr_center_y - prev_center_y)**2
            )
            total_distance += distance
        
        if self.pixel_to_meter:
            total_distance *= self.pixel_to_meter
        
        return total_distance
    
    def calculate_average_speed(self, track: Track) -> float:
        """Calculate average speed"""
        total_distance = self.calculate_total_distance(track)
        if track.age == 0:
            return 0.0
        
        avg_speed = total_distance / track.age
        
        if self.pixel_to_meter:
            avg_speed *= self.fps  # Convert to m/s
        
        return avg_speed
    
    def smooth_trajectory(
        self,
        track: Track,
        window_size: int = 5
    ) -> List[Tuple[float, float, float, float]]:
        """
        Smooth trajectory using moving average
        
        Args:
            track: Track object
            window_size: Size of moving average window
            
        Returns:
            List of smoothed bounding boxes
        """
        if len(track.history) < window_size:
            return track.history
        
        smoothed = []
        for i in range(len(track.history)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(track.history), i + window_size // 2 + 1)
            window = track.history[start_idx:end_idx]
            
            avg_box = (
                np.mean([b[0] for b in window]),
                np.mean([b[1] for b in window]),
                np.mean([b[2] for b in window]),
                np.mean([b[3] for b in window])
            )
            smoothed.append(avg_box)
        
        return smoothed
    
    def predict_next_position(
        self,
        track: Track,
        frames_ahead: int = 1
    ) -> Tuple[float, float, float, float]:
        """
        Predict next position using linear extrapolation
        
        Args:
            track: Track object
            frames_ahead: Number of frames to predict ahead
            
        Returns:
            Predicted bounding box
        """
        if len(track.history) < 2:
            return track.bbox
        
        # Calculate average velocity
        velocities = []
        for i in range(1, len(track.history)):
            prev_bbox = track.history[i-1]
            curr_bbox = track.history[i]
            
            vx = (curr_bbox[0] - prev_bbox[0] + curr_bbox[2] - prev_bbox[2]) / 2
            vy = (curr_bbox[1] - prev_bbox[1] + curr_bbox[3] - prev_bbox[3]) / 2
            velocities.append((vx, vy))
        
        avg_vx = np.mean([v[0] for v in velocities])
        avg_vy = np.mean([v[1] for v in velocities])
        
        # Predict
        x1, y1, x2, y2 = track.bbox
        w = x2 - x1
        h = y2 - y1
        
        predicted = (
            x1 + avg_vx * frames_ahead,
            y1 + avg_vy * frames_ahead,
            x2 + avg_vx * frames_ahead,
            y2 + avg_vy * frames_ahead
        )
        
        return predicted
    
    def analyze_track(self, track: Track) -> Dict[str, Any]:
        """Get comprehensive analysis of a track"""
        speed_x, speed_y = self.calculate_speed(track)
        speed_magnitude = np.sqrt(speed_x**2 + speed_y**2)
        direction = self.calculate_direction(track)
        total_distance = self.calculate_total_distance(track)
        avg_speed = self.calculate_average_speed(track)
        
        return {
            "track_id": track.track_id,
            "speed": {
                "x": speed_x,
                "y": speed_y,
                "magnitude": speed_magnitude
            },
            "direction": direction,
            "total_distance": total_distance,
            "average_speed": avg_speed,
            "trajectory_length": len(track.history),
            "age": track.age
        }
    
    def analyze_tracks(self, tracks: List[Track]) -> List[Dict[str, Any]]:
        """Analyze multiple tracks"""
        return [self.analyze_track(track) for track in tracks]

