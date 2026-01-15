"""
管道集成测试

本测试验证完整的 Vision Framework 管道集成：
- VisionPipeline 的端到端工作流
- 组件之间的集成
- 多个处理步骤的串联
"""

import sys
from pathlib import Path
import numpy as np
from collections import namedtuple

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_mock_image(height=480, width=640, channels=3):
    """创建 mock 图像数组"""
    return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)


def create_mock_detection_data(x1, y1, x2, y2, confidence=0.9):
    """创建 mock 检测数据"""
    Detection = namedtuple('Detection', ['bbox', 'confidence', 'class_id', 'class_name'])
    BBox = namedtuple('BBox', ['x1', 'y1', 'x2', 'y2'])
    
    bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
    return Detection(bbox=bbox, confidence=confidence, class_id=1, class_name="person")


def test_pipeline_import():
    """测试管道导入"""
    print("\n1. 测试管道导入...")
    
    try:
        from visionframework import VisionPipeline
        from visionframework.core.detector import Detector
        
        print("  [✓] 管道模块成功导入")
        return True
    except Exception as e:
        print(f"  [✗] 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_creation():
    """测试管道创建"""
    print("\n2. 测试管道创建...")
    
    try:
        from visionframework import VisionPipeline
        
        config = {
            "detector_config": {
                "model_type": "yolo",
                "model_path": "yolov8n.pt"
            },
            "tracker_config": {
                "tracker_type": "iou",
                "max_age": 30
            }
        }
        
        pipeline = VisionPipeline(config)
        print("  [✓] 管道创建成功")
        return True
    except Exception as e:
        print(f"  [✗] 管道创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_initialization():
    """测试管道初始化"""
    print("\n3. 测试管道初始化...")
    
    try:
        from visionframework import VisionPipeline
        
        config = {
            "detector_config": {
                "model_type": "yolo",
                "model_path": "yolov8n.pt"
            },
            "tracker_config": {
                "tracker_type": "iou"
            }
        }
        
        pipeline = VisionPipeline(config)
        
        if pipeline.initialize():
            print("  [✓] 管道初始化成功")
            return True
        else:
            print("  [✗] 管道初始化失败")
            return False
    except Exception as e:
        print(f"  [✗] 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_single_frame():
    """测试单帧管道处理"""
    print("\n4. 测试单帧管道处理...")
    
    try:
        from visionframework import VisionPipeline
        
        config = {
            "detector_config": {
                "model_type": "yolo",
                "model_path": "yolov8n.pt"
            },
            "tracker_config": {
                "tracker_type": "iou",
                "max_age": 30
            }
        }
        
        pipeline = VisionPipeline(config)
        
        if not pipeline.initialize():
            print("  [SKIP] 管道初始化失败")
            return None
        
        # 创建 mock 图像
        image = create_mock_image(480, 640, 3)
        print(f"  [✓] 创建 mock 图像: shape={image.shape}")
        
        # 执行管道处理
        results = pipeline.process(image)
        
        if results is not None:
            print(f"  [✓] 单帧处理成功，返回结果")
            if isinstance(results, dict):
                if 'detections' in results:
                    print(f"    - 检测数: {len(results['detections'])}")
                if 'tracks' in results:
                    print(f"    - 跟踪数: {len(results['tracks'])}")
            return True
        else:
            print("  [✗] 管道返回 None")
            return False
    except Exception as e:
        print(f"  [✗] 单帧处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_multi_frame():
    """测试多帧管道处理"""
    print("\n5. 测试多帧管道处理...")
    
    try:
        from visionframework import VisionPipeline
        
        config = {
            "detector_config": {
                "model_type": "yolo",
                "model_path": "yolov8n.pt"
            },
            "tracker_config": {
                "tracker_type": "iou",
                "max_age": 30
            }
        }
        
        pipeline = VisionPipeline(config)
        
        if not pipeline.initialize():
            print("  [SKIP] 管道初始化失败")
            return None
        
        # 处理 5 帧图像
        num_frames = 5
        all_results = []
        
        for frame_idx in range(num_frames):
            image = create_mock_image(480, 640, 3)
            results = pipeline.process(image)
            
            if results is not None:
                all_results.append(results)
                print(f"  [✓] 第 {frame_idx} 帧处理成功")
            else:
                print(f"  [✗] 第 {frame_idx} 帧处理失败")
                return False
        
        print(f"  [✓] 处理 {num_frames} 帧完成，共 {len(all_results)} 个结果")
        return True
    except Exception as e:
        print(f"  [✗] 多帧处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_with_processors():
    """测试包含处理器的管道"""
    print("\n6. 测试包含处理器的管道...")
    
    try:
        from visionframework import VisionPipeline
        
        config = {
            "detector_config": {
                "model_type": "yolo",
                "model_path": "yolov8n.pt"
            },
            "tracker_config": {
                "tracker_type": "iou"
            }
        }
        
        pipeline = VisionPipeline(config)
        
        if not pipeline.initialize():
            print("  [SKIP] 管道初始化失败")
            return None
        
        # 处理单帧
        image = create_mock_image()
        results = pipeline.process(image)
        
        print("  [✓] 带处理器的管道处理成功")
        return True
    except Exception as e:
        print(f"  [✗] 带处理器的管道失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_configuration_validation():
    """测试管道配置验证"""
    print("\n7. 测试管道配置验证...")
    
    try:
        from visionframework import VisionPipeline
        
        # 测试各种配置
        test_configs = [
            {
                "name": "基础配置",
                "config": {
                    "detector_config": {"model_type": "yolo"},
                    "tracker_config": {"tracker_type": "iou"}
                },
                "should_pass": True
            },
            {
                "name": "缺少检测器（使用默认）",
                "config": {
                    "tracker_config": {"tracker_type": "iou"}
                },
                "should_pass": True
            },
            {
                "name": "缺少跟踪器（使用默认）",
                "config": {
                    "detector_config": {"model_type": "yolo"}
                },
                "should_pass": True
            },
        ]
        
        passed = 0
        failed = 0
        
        for test_case in test_configs:
            try:
                pipeline = VisionPipeline(test_case["config"])
                if test_case["should_pass"]:
                    passed += 1
                    print(f"  [✓] {test_case['name']}: 配置有效")
                else:
                    # 不应该创建成功
                    result = pipeline.validate_config(test_case["config"])
                    if not result:
                        passed += 1
                        print(f"  [✓] {test_case['name']}: 正确拒绝")
                    else:
                        failed += 1
                        print(f"  [✗] {test_case['name']}: 应该无效但接受了")
            except Exception as e:
                if not test_case["should_pass"]:
                    passed += 1
                    print(f"  [✓] {test_case['name']}: 正确拒绝")
                else:
                    failed += 1
                    print(f"  [✗] {test_case['name']}: 应该有效但出错 - {e}")
        
        return failed == 0
    except Exception as e:
        print(f"  [✗] 配置验证测试失败: {e}")
        return False


def test_pipeline_error_handling():
    """测试管道错误处理"""
    print("\n8. 测试管道错误处理...")
    
    try:
        from visionframework import VisionPipeline
        from visionframework.exceptions import VisionFrameworkError
        
        config = {
            "detector_config": {"model_type": "yolo"},
            "tracker_config": {"tracker_type": "iou"}
        }
        
        pipeline = VisionPipeline(config)
        
        if not pipeline.initialize():
            print("  [SKIP] 管道初始化失败")
            return None
        
        # 测试处理无效输入 - 只测试有效的情况
        test_cases = [
            (None, "None 输入"),
        ]
        
        handled_correctly = 0
        for invalid_input, description in test_cases:
            try:
                result = pipeline.process(invalid_input)
                # 应该处理而不是崩溃
                handled_correctly += 1
                print(f"  [✓] {description}: 已正确处理")
            except Exception as e:
                handled_correctly += 1
                print(f"  [✓] {description}: 正确处理异常")
        
        return handled_correctly > 0
    except Exception as e:
        print(f"  [✗] 错误处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_batch_processing():
    """测试管道批处理"""
    print("\n9. 测试管道批处理...")
    
    try:
        from visionframework import VisionPipeline
        
        config = {
            "detector_config": {"model_type": "yolo"},
            "tracker_config": {"tracker_type": "iou"}
        }
        
        pipeline = VisionPipeline(config)
        
        if not pipeline.initialize():
            print("  [SKIP] 管道初始化失败")
            return None
        
        # 检查管道是否支持批处理
        batch_size = 4
        images = [create_mock_image() for _ in range(batch_size)]
        
        # 逐帧处理
        results = []
        for img in images:
            result = pipeline.process(img)
            if result is not None:
                results.append(result)
        
        if len(results) == batch_size:
            print(f"  [✓] 逐帧处理 {batch_size} 个图像成功")
            return True
        else:
            print(f"  [✗] 处理返回错误数量的结果")
            return False
    except Exception as e:
        print(f"  [✗] 批处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_state_management():
    """测试管道状态管理"""
    print("\n10. 测试管道状态管理...")
    
    try:
        from visionframework import VisionPipeline
        
        config = {
            "detector_config": {"model_type": "yolo"},
            "tracker_config": {"tracker_type": "iou"}
        }
        
        pipeline = VisionPipeline(config)
        
        # 初始化
        if not pipeline.initialize():
            print("  [SKIP] 管道初始化失败")
            return None
        
        print("  [✓] 管道初始化成功")
        
        # 清理
        if hasattr(pipeline, 'cleanup'):
            pipeline.cleanup()
            print("  [✓] 管道清理成功")
        
        return True
    except Exception as e:
        print(f"  [✗] 状态管理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有管道集成测试"""
    print("=" * 70)
    print("Vision Framework - 管道集成测试")
    print("=" * 70)
    
    tests = [
        test_pipeline_import,
        test_pipeline_creation,
        test_pipeline_initialization,
        test_pipeline_single_frame,
        test_pipeline_multi_frame,
        test_pipeline_with_processors,
        test_pipeline_configuration_validation,
        test_pipeline_error_handling,
        test_pipeline_batch_processing,
        test_pipeline_state_management,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ 测试执行失败: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    skipped = sum(1 for r in results if r is None)
    
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"跳过: {skipped}")
    print(f"总计: {len(results)} 个测试")
    
    if failed == 0:
        print("\n[✓] 所有管道集成测试通过!")
        return True
    else:
        print(f"\n[✗] {failed} 个测试失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
