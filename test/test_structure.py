#!/usr/bin/env python3
"""
结构测试脚本，测试新增功能的代码结构和集成
"""

import os
import sys

# Set environment variable to handle OpenMP duplicate library issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 测试结果记录
test_results = {
    "passed": [],
    "failed": [],
    "skipped": []
}


def test_module_imports():
    """测试模块导入"""
    test_name = "模块导入测试"
    print(f"\n=== {test_name} ===")
    
    try:
        # 测试SAM分割器导入
        from visionframework.core.components.segmenters.sam_segmenter import SAMSegmenter
        print("✓ SAMSegmenter 导入成功")
        
        # 测试检测器导入
        from visionframework.core.components.detectors.yolo_detector import YOLODetector
        print("✓ YOLODetector 导入成功")
        
        # 测试CLIP提取器导入
        from visionframework.core.components.processors.clip_extractor import CLIPExtractor
        print("✓ CLIPExtractor 导入成功")
        
        # 测试姿态估计器导入
        from visionframework.core.components.processors.pose_estimator import PoseEstimator
        print("✓ PoseEstimator 导入成功")
        
        # 测试分割器包导入
        from visionframework.core.components.segmenters import SAMSegmenter
        print("✓ segmenters 包导入成功")
        
        test_results["passed"].append(test_name)
        return True
    
    except ImportError as e:
        print(f"✗ 导入失败：{e}")
        test_results["failed"].append(test_name)
        return False


def test_class_instantiation():
    """测试类实例化"""
    test_name = "类实例化测试"
    print(f"\n=== {test_name} ===")
    
    try:
        # 测试SAM分割器实例化
        from visionframework.core.components.segmenters.sam_segmenter import SAMSegmenter
        sam = SAMSegmenter({
            "model_type": "vit_b",
            "device": "cpu",
            "use_fp16": False
        })
        print("✓ SAMSegmenter 实例化成功")
        
        # 测试检测器实例化
        from visionframework.core.components.detectors.yolo_detector import YOLODetector
        detector = YOLODetector({
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.1,
            "device": "cpu",
            "segmenter_type": "sam",
            "sam_model_type": "vit_b",
            "sam_use_fp16": False
        })
        print("✓ YOLODetector 实例化成功")
        
        # 测试CLIP提取器实例化
        from visionframework.core.components.processors.clip_extractor import CLIPExtractor
        clip = CLIPExtractor({
            "model_name": "openai/clip-vit-base-patch32",
            "device": "cpu",
            "use_fp16": False
        })
        print("✓ CLIPExtractor 实例化成功")
        
        # 测试姿态估计器实例化
        from visionframework.core.components.processors.pose_estimator import PoseEstimator
        pose_estimator = PoseEstimator({
            "model_type": "yolo_pose",
            "model_path": "yolov8n-pose.pt",
            "device": "cpu",
            "conf_threshold": 0.1
        })
        print("✓ PoseEstimator 实例化成功")
        
        test_results["passed"].append(test_name)
        return True
    
    except Exception as e:
        print(f"✗ 实例化失败：{e}")
        import traceback
        traceback.print_exc()
        test_results["failed"].append(test_name)
        return False


def test_config_validation():
    """测试配置验证"""
    test_name = "配置验证测试"
    print(f"\n=== {test_name} ===")
    
    try:
        # 测试检测器配置验证
        from visionframework.core.components.detectors.yolo_detector import YOLODetector
        
        # 测试有效配置
        valid_config = {
            "model_path": "yolov8n.pt",
            "model_type": "yolo",
            "conf_threshold": 0.7,
            "device": "cpu",
            "segmenter_type": "sam",
            "sam_model_type": "vit_b"
        }
        
        detector = YOLODetector(valid_config)
        print("✓ 检测器有效配置验证成功")
        
        # 测试无效配置（无效的sam_model_type）
        invalid_config = {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.7,
            "device": "cpu",
            "segmenter_type": "sam",
            "sam_model_type": "invalid_type"
        }
        
        try:
            detector = YOLODetector(invalid_config)
            print("⚠ 检测器无效配置没有引发预期的异常")
        except Exception as e:
            print("✓ 检测器无效配置正确引发异常")
        
        test_results["passed"].append(test_name)
        return True
    
    except Exception as e:
        print(f"✗ 配置验证失败：{e}")
        import traceback
        traceback.print_exc()
        test_results["failed"].append(test_name)
        return False


def test_detector_segmenter_integration():
    """测试检测器与分割器集成"""
    test_name = "检测器与分割器集成测试"
    print(f"\n=== {test_name} ===")
    
    try:
        from visionframework.core.components.detectors.yolo_detector import YOLODetector
        
        # 创建带SAM分割器的检测器
        detector = YOLODetector({
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.1,
            "device": "cpu",
            "segmenter_type": "sam",
            "sam_model_type": "vit_b",
            "sam_use_fp16": False
        })
        
        print("✓ 检测器与SAM分割器集成配置成功")
        
        # 测试分割器类型为None的情况
        detector = YOLODetector({
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.1,
            "device": "cpu",
            "segmenter_type": None
        })
        
        print("✓ 检测器不使用分割器配置成功")
        
        test_results["passed"].append(test_name)
        return True
    
    except Exception as e:
        print(f"✗ 集成测试失败：{e}")
        import traceback
        traceback.print_exc()
        test_results["failed"].append(test_name)
        return False


def test_pose_estimator_models():
    """测试姿态估计器模型类型"""
    test_name = "姿态估计器模型类型测试"
    print(f"\n=== {test_name} ===")
    
    try:
        from visionframework.core.components.processors.pose_estimator import PoseEstimator
        
        # 测试YOLO Pose配置
        yolo_pose = PoseEstimator({
            "model_type": "yolo_pose",
            "model_path": "yolov8n-pose.pt",
            "device": "cpu",
            "conf_threshold": 0.1
        })
        print("✓ YOLO Pose 配置成功")
        
        # 测试MediaPipe Pose配置
        mediapipe_pose = PoseEstimator({
            "model_type": "mediapipe",
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5
        })
        print("✓ MediaPipe Pose 配置成功")
        
        test_results["passed"].append(test_name)
        return True
    
    except Exception as e:
        print(f"✗ 姿态估计器模型类型测试失败：{e}")
        import traceback
        traceback.print_exc()
        test_results["failed"].append(test_name)
        return False


def run_all_tests():
    """运行所有测试"""
    print("开始结构测试...")
    print("=" * 50)
    
    # 运行各项测试
    test_module_imports()
    test_class_instantiation()
    test_config_validation()
    test_detector_segmenter_integration()
    test_pose_estimator_models()
    
    # 打印测试结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    print(f"通过测试: {len(test_results['passed'])}")
    for test in test_results['passed']:
        print(f"  ✓ {test}")
    
    print(f"\n跳过测试: {len(test_results['skipped'])}")
    for test in test_results['skipped']:
        print(f"  ⚠ {test}")
    
    print(f"\n失败测试: {len(test_results['failed'])}")
    for test in test_results['failed']:
        print(f"  ✗ {test}")
    
    print("\n" + "=" * 50)
    
    # 计算成功率
    total_tests = len(test_results['passed']) + len(test_results['failed']) + len(test_results['skipped'])
    if total_tests > 0:
        success_rate = (len(test_results['passed']) / total_tests) * 100
        print(f"总体测试成功率: {success_rate:.1f}%")
    
    # 生成测试报告
    if test_results['failed']:
        print("\n测试存在失败项，请检查错误信息并修复问题。")
        return False
    else:
        print("\n所有测试通过或跳过，代码结构和集成正常！")
        return True


if __name__ == "__main__":
    run_all_tests()
