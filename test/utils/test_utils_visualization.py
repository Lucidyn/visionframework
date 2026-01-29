"""
测试可视化工具类
"""

import numpy as np
import pytest
from visionframework.utils.visualization import (
    BaseVisualizer,
    DetectionVisualizer,
    TrackVisualizer,
    PoseVisualizer,
    Visualizer
)
from visionframework.data.detection import Detection
from visionframework.data.track import Track
from visionframework.data.pose import Pose, KeyPoint


# 创建测试图像
def create_test_image():
    """创建测试图像"""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# 测试BaseVisualizer
def test_base_visualizer():
    """测试基础可视化器"""
    viz = BaseVisualizer()
    
    # 测试颜色生成（使用实例的color_palette）
    assert len(viz.color_palette) == 100  # 默认生成100种颜色
    
    # 测试获取颜色
    color1 = viz._get_color(0)
    color2 = viz._get_color(1)
    color3 = viz._get_color(100)  # 测试循环，100 % 100 = 0
    color4 = viz._get_color(150)  # 测试循环，150 % 100 = 50
    
    assert isinstance(color1, tuple)
    assert len(color1) == 3
    assert color1 != color2
    assert color3 == color1  # 应该相等，因为100 % 100 = 0
    assert color4 != color1  # 应该不相等，因为50 % 100 = 50
    
    # 测试生成不同数量的颜色
    viz2 = BaseVisualizer({"color_count": 20})
    color_palette2 = viz2._generate_color_palette(20)
    assert len(color_palette2) == 20


# 测试DetectionVisualizer
def test_detection_visualizer():
    """测试检测可视化器"""
    image = create_test_image()
    viz = DetectionVisualizer()
    
    # 创建测试检测
    sample_detection = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.95,
        class_id=0,
        class_name="person"
    )
    
    sample_detections = [
        sample_detection,
        Detection(
            bbox=(300, 300, 400, 400),
            confidence=0.85,
            class_id=1,
            class_name="car"
        )
    ]
    
    # 测试绘制单个检测
    result = viz.draw_detection(image.copy(), sample_detection)
    assert result.shape == image.shape
    
    # 测试绘制多个检测
    result = viz.draw_detections(image.copy(), sample_detections)
    assert result.shape == image.shape
    
    # 测试配置选项
    viz = DetectionVisualizer({"show_labels": False, "show_confidences": False})
    result = viz.draw_detection(image.copy(), sample_detection)
    assert result.shape == image.shape


# 测试TrackVisualizer
def test_track_visualizer():
    """测试跟踪可视化器"""
    image = create_test_image()
    viz = TrackVisualizer()
    
    # 创建测试跟踪
    sample_track = Track(
        track_id=1,
        bbox=(100, 100, 200, 200),
        confidence=0.95,
        class_id=0,
        class_name="person"
    )
    
    # 添加历史记录
    for i in range(5):
        sample_track.update(
            bbox=(100 + i * 5, 100 + i * 5, 200 + i * 5, 200 + i * 5),
            confidence=0.95
        )
    
    sample_tracks = [
        sample_track,
        Track(
            track_id=2,
            bbox=(300, 300, 400, 400),
            confidence=0.85,
            class_id=1,
            class_name="car"
        )
    ]
    
    # 测试绘制单个跟踪
    result = viz.draw_track(image.copy(), sample_track, draw_history=True)
    assert result.shape == image.shape
    
    # 测试绘制多个跟踪
    result = viz.draw_tracks(image.copy(), sample_tracks, draw_history=True)
    assert result.shape == image.shape
    
    # 测试不绘制历史记录
    result = viz.draw_track(image.copy(), sample_track, draw_history=False)
    assert result.shape == image.shape


# 测试PoseVisualizer
def test_pose_visualizer():
    """测试姿态可视化器"""
    image = create_test_image()
    viz = PoseVisualizer()
    
    # 创建测试姿态
    keypoints = [
        KeyPoint(keypoint_id=0, x=150, y=150, confidence=0.9, keypoint_name="nose"),
        KeyPoint(keypoint_id=1, x=140, y=140, confidence=0.8, keypoint_name="left_eye"),
        KeyPoint(keypoint_id=2, x=160, y=140, confidence=0.8, keypoint_name="right_eye"),
        KeyPoint(keypoint_id=3, x=130, y=160, confidence=0.7, keypoint_name="left_ear"),
        KeyPoint(keypoint_id=4, x=170, y=160, confidence=0.7, keypoint_name="right_ear"),
        KeyPoint(keypoint_id=5, x=120, y=200, confidence=0.8, keypoint_name="left_shoulder"),
        KeyPoint(keypoint_id=6, x=180, y=200, confidence=0.8, keypoint_name="right_shoulder"),
        KeyPoint(keypoint_id=7, x=110, y=250, confidence=0.7, keypoint_name="left_elbow"),
        KeyPoint(keypoint_id=8, x=190, y=250, confidence=0.7, keypoint_name="right_elbow"),
        KeyPoint(keypoint_id=9, x=100, y=300, confidence=0.6, keypoint_name="left_wrist"),
        KeyPoint(keypoint_id=10, x=200, y=300, confidence=0.6, keypoint_name="right_wrist"),
        KeyPoint(keypoint_id=11, x=130, y=350, confidence=0.7, keypoint_name="left_hip"),
        KeyPoint(keypoint_id=12, x=170, y=350, confidence=0.7, keypoint_name="right_hip"),
        KeyPoint(keypoint_id=13, x=120, y=400, confidence=0.6, keypoint_name="left_knee"),
        KeyPoint(keypoint_id=14, x=180, y=400, confidence=0.6, keypoint_name="right_knee"),
        KeyPoint(keypoint_id=15, x=110, y=450, confidence=0.5, keypoint_name="left_ankle"),
        KeyPoint(keypoint_id=16, x=190, y=450, confidence=0.5, keypoint_name="right_ankle"),
    ]
    
    sample_pose = Pose(
        bbox=(100, 100, 200, 480),
        confidence=0.9,
        pose_id=1,
        keypoints=keypoints
    )
    
    # 测试绘制单个姿态
    result = viz.draw_pose(
        image.copy(), 
        sample_pose, 
        draw_skeleton=True, 
        draw_keypoints=True, 
        draw_bbox=True
    )
    assert result.shape == image.shape
    
    # 测试只绘制关键点
    result = viz.draw_pose(
        image.copy(), 
        sample_pose, 
        draw_skeleton=False, 
        draw_keypoints=True, 
        draw_bbox=False
    )
    assert result.shape == image.shape
    
    # 测试只绘制骨骼
    result = viz.draw_pose(
        image.copy(), 
        sample_pose, 
        draw_skeleton=True, 
        draw_keypoints=False, 
        draw_bbox=False
    )
    assert result.shape == image.shape


# 测试统一可视化器
def test_visualizer():
    """测试统一可视化器"""
    image = create_test_image()
    viz = Visualizer()
    
    # 创建测试数据
    sample_detection = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.95,
        class_id=0,
        class_name="person"
    )
    
    sample_detections = [
        sample_detection,
        Detection(
            bbox=(300, 300, 400, 400),
            confidence=0.85,
            class_id=1,
            class_name="car"
        )
    ]
    
    sample_track = Track(
        track_id=1,
        bbox=(100, 100, 200, 200),
        confidence=0.95,
        class_id=0,
        class_name="person"
    )
    
    for i in range(5):
        sample_track.update(
            bbox=(100 + i * 5, 100 + i * 5, 200 + i * 5, 200 + i * 5),
            confidence=0.95
        )
    
    sample_tracks = [sample_track]
    
    keypoints = [
        KeyPoint(keypoint_id=0, x=150, y=150, confidence=0.9, keypoint_name="nose"),
        KeyPoint(keypoint_id=1, x=140, y=140, confidence=0.8, keypoint_name="left_eye"),
        KeyPoint(keypoint_id=2, x=160, y=140, confidence=0.8, keypoint_name="right_eye"),
    ]
    
    sample_pose = Pose(
        bbox=(100, 100, 200, 300),
        confidence=0.9,
        pose_id=1,
        keypoints=keypoints
    )
    
    # 测试绘制检测结果
    result = viz.draw_detections(image.copy(), sample_detections)
    assert result.shape == image.shape
    
    # 测试绘制跟踪结果
    result = viz.draw_tracks(image.copy(), sample_tracks)
    assert result.shape == image.shape
    
    # 测试绘制姿态结果
    result = viz.draw_poses(image.copy(), [sample_pose])
    assert result.shape == image.shape
    
    # 测试绘制多种结果
    result = viz.draw_results(
        image.copy(),
        detections=sample_detections,
        tracks=None,
        poses=[sample_pose]
    )
    assert result.shape == image.shape
    
    # 测试同时绘制检测和跟踪（应该只显示跟踪）
    result = viz.draw_results(
        image.copy(),
        detections=sample_detections,
        tracks=sample_tracks,
        poses=None
    )
    assert result.shape == image.shape
