"""
测试评估工具类
"""

import numpy as np
import pytest
from visionframework.utils.evaluation.detection_evaluator import DetectionEvaluator
from visionframework.utils.evaluation.tracking_evaluator import TrackingEvaluator
from visionframework.data.detection import Detection


# 测试DetectionEvaluator类
def test_detection_evaluator_initialization():
    """测试检测评估器初始化"""
    evaluator = DetectionEvaluator()
    assert evaluator.iou_threshold == 0.5
    
    evaluator = DetectionEvaluator(iou_threshold=0.75)
    assert evaluator.iou_threshold == 0.75


def test_detection_evaluator_iou_calculation():
    """测试IoU计算"""
    evaluator = DetectionEvaluator()
    
    # 完全重叠的两个框
    iou1 = evaluator._calculate_iou((0, 0, 100, 100), (0, 0, 100, 100))
    assert iou1 == 1.0
    
    # 部分重叠的两个框
    iou2 = evaluator._calculate_iou((0, 0, 100, 100), (50, 50, 150, 150))
    assert 0 < iou2 < 1.0
    
    # 完全不重叠的两个框
    iou3 = evaluator._calculate_iou((0, 0, 100, 100), (200, 200, 300, 300))
    assert iou3 == 0.0


def test_detection_evaluator_match_detections():
    """测试检测匹配"""
    evaluator = DetectionEvaluator(iou_threshold=0.5)
    
    # 创建预测检测
    pred_detections = [
        Detection(bbox=(0, 0, 100, 100), confidence=0.95, class_id=0, class_name="person"),
        Detection(bbox=(200, 200, 300, 300), confidence=0.9, class_id=1, class_name="car")
    ]
    
    # 创建真实检测
    gt_detections = [
        Detection(bbox=(10, 10, 110, 110), confidence=1.0, class_id=0, class_name="person"),
        Detection(bbox=(210, 210, 310, 310), confidence=1.0, class_id=1, class_name="car")
    ]
    
    matched_pred, matched_gt, matches = evaluator.match_detections(pred_detections, gt_detections)
    assert len(matches) == 2
    assert len(matched_pred) == 2
    assert len(matched_gt) == 2


def test_detection_evaluator_calculate_metrics():
    """测试检测指标计算"""
    evaluator = DetectionEvaluator(iou_threshold=0.5)
    
    # 创建预测检测
    pred_detections = [
        Detection(bbox=(0, 0, 100, 100), confidence=0.95, class_id=0, class_name="person"),
        Detection(bbox=(200, 200, 300, 300), confidence=0.9, class_id=1, class_name="car"),
        Detection(bbox=(400, 400, 500, 500), confidence=0.8, class_id=2, class_name="bike")  # FP
    ]
    
    # 创建真实检测
    gt_detections = [
        Detection(bbox=(10, 10, 110, 110), confidence=1.0, class_id=0, class_name="person"),
        Detection(bbox=(210, 210, 310, 310), confidence=1.0, class_id=1, class_name="car"),
        Detection(bbox=(600, 600, 700, 700), confidence=1.0, class_id=2, class_name="bike")  # FN
    ]
    
    metrics = evaluator.calculate_metrics(pred_detections, gt_detections)
    
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "tp" in metrics
    assert "fp" in metrics
    assert "fn" in metrics
    
    # 应该有2个TP，1个FP，1个FN
    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1


def test_detection_evaluator_calculate_map():
    """测试mAP计算"""
    evaluator = DetectionEvaluator(iou_threshold=0.5)
    
    # 创建多帧预测和真实检测
    all_pred_detections = [
        [
            Detection(bbox=(0, 0, 100, 100), confidence=0.95, class_id=0, class_name="person"),
            Detection(bbox=(200, 200, 300, 300), confidence=0.9, class_id=1, class_name="car")
        ],
        [
            Detection(bbox=(50, 50, 150, 150), confidence=0.85, class_id=0, class_name="person"),
            Detection(bbox=(250, 250, 350, 350), confidence=0.8, class_id=1, class_name="car")
        ]
    ]
    
    all_gt_detections = [
        [
            Detection(bbox=(10, 10, 110, 110), confidence=1.0, class_id=0, class_name="person"),
            Detection(bbox=(210, 210, 310, 310), confidence=1.0, class_id=1, class_name="car")
        ],
        [
            Detection(bbox=(60, 60, 160, 160), confidence=1.0, class_id=0, class_name="person"),
            Detection(bbox=(260, 260, 360, 360), confidence=1.0, class_id=1, class_name="car")
        ]
    ]
    
    map_result = evaluator.calculate_map(all_pred_detections, all_gt_detections)
    
    assert "mAP" in map_result
    assert "AP_per_class" in map_result
    assert "num_classes" in map_result
    
    # 允许小的浮点数误差
    assert 0 <= map_result["mAP"] <= 1.0001
    assert len(map_result["AP_per_class"]) == map_result["num_classes"]


# 测试TrackingEvaluator类
def test_tracking_evaluator_initialization():
    """测试跟踪评估器初始化"""
    evaluator = TrackingEvaluator()
    assert evaluator.iou_threshold == 0.5
    
    evaluator = TrackingEvaluator(iou_threshold=0.7)
    assert evaluator.iou_threshold == 0.7


def test_tracking_evaluator_iou_calculation():
    """测试跟踪评估器的IoU计算"""
    evaluator = TrackingEvaluator()
    
    # 测试IoU计算
    iou = evaluator._compute_iou(
        {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100},
        {'x1': 50, 'y1': 50, 'x2': 150, 'y2': 150}
    )
    assert 0 < iou < 1.0


def test_tracking_evaluator_match_detections():
    """测试跟踪检测匹配"""
    evaluator = TrackingEvaluator()
    
    # 创建预测框和真实框
    pred_boxes = [
        {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100},
        {'x1': 200, 'y1': 200, 'x2': 300, 'y2': 300}
    ]
    
    gt_boxes = [
        {'x1': 10, 'y1': 10, 'x2': 110, 'y2': 110},
        {'x1': 210, 'y1': 210, 'x2': 310, 'y2': 310}
    ]
    
    matched_pairs, unmatched_pred, unmatched_gt = evaluator._match_detections(pred_boxes, gt_boxes)
    assert len(matched_pairs) == 2
    assert len(unmatched_pred) == 0
    assert len(unmatched_gt) == 0


def test_tracking_evaluator_mota_calculation():
    """测试MOTA计算"""
    evaluator = TrackingEvaluator()
    
    # 创建简单的跟踪数据
    pred_tracks = [
        [{'track_id': 1, 'bbox': {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100}}],
        [{'track_id': 1, 'bbox': {'x1': 10, 'y1': 10, 'x2': 110, 'y2': 110}}],
        [{'track_id': 2, 'bbox': {'x1': 20, 'y1': 20, 'x2': 120, 'y2': 120}}]  # ID切换
    ]
    
    gt_tracks = [
        [{'track_id': 1, 'bbox': {'x1': 5, 'y1': 5, 'x2': 105, 'y2': 105}}],
        [{'track_id': 1, 'bbox': {'x1': 15, 'y1': 15, 'x2': 115, 'y2': 115}}],
        [{'track_id': 1, 'bbox': {'x1': 25, 'y1': 25, 'x2': 125, 'y2': 125}}]
    ]
    
    mota_result = evaluator.calculate_mota(pred_tracks, gt_tracks)
    assert "MOTA" in mota_result
    assert "precision" in mota_result
    assert "recall" in mota_result
    
    # MOTA应该在0到1之间，包括0和1
    assert 0 <= mota_result["MOTA"] <= 1.0


def test_tracking_evaluator_motp_calculation():
    """测试MOTP计算"""
    evaluator = TrackingEvaluator()
    
    # 创建简单的跟踪数据
    pred_tracks = [
        [{'track_id': 1, 'bbox': {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100}}],
        [{'track_id': 1, 'bbox': {'x1': 10, 'y1': 10, 'x2': 110, 'y2': 110}}]
    ]
    
    gt_tracks = [
        [{'track_id': 1, 'bbox': {'x1': 5, 'y1': 5, 'x2': 105, 'y2': 105}}],
        [{'track_id': 1, 'bbox': {'x1': 15, 'y1': 15, 'x2': 115, 'y2': 115}}]
    ]
    
    motp_result = evaluator.calculate_motp(pred_tracks, gt_tracks)
    assert "MOTP" in motp_result
    assert "total_matched_pairs" in motp_result
    assert "total_distance" in motp_result


def test_tracking_evaluator_idf1_calculation():
    """测试IDF1计算"""
    evaluator = TrackingEvaluator()
    
    # 创建简单的跟踪数据
    pred_tracks = [
        [{'track_id': 1, 'bbox': {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100}}],
        [{'track_id': 1, 'bbox': {'x1': 10, 'y1': 10, 'x2': 110, 'y2': 110}}],
        [{'track_id': 1, 'bbox': {'x1': 20, 'y1': 20, 'x2': 120, 'y2': 120}}]
    ]
    
    gt_tracks = [
        [{'track_id': 1, 'bbox': {'x1': 5, 'y1': 5, 'x2': 105, 'y2': 105}}],
        [{'track_id': 1, 'bbox': {'x1': 15, 'y1': 15, 'x2': 115, 'y2': 115}}],
        [{'track_id': 1, 'bbox': {'x1': 25, 'y1': 25, 'x2': 125, 'y2': 125}}]
    ]
    
    idf1_result = evaluator.calculate_idf1(pred_tracks, gt_tracks)
    assert "IDF1" in idf1_result
    assert "IDTP" in idf1_result
    assert "IDFP" in idf1_result
    assert "IDFN" in idf1_result
    
    # 所有ID都匹配，IDF1应该较高
    assert 0.5 <= idf1_result["IDF1"] <= 1.0
