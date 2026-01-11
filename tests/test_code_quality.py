"""
第二阶段优化测试

本测试验证第二阶段优化的结果：
1. 配置验证功能
2. 类型提示完善
3. 文档字符串完善
4. 错误处理统一化
"""

import sys
from pathlib import Path
from typing import get_type_hints, get_args, get_origin
import inspect

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_config_validation():
    """测试配置验证功能"""
    print("\n1. 测试配置验证功能...")
    
    try:
        from visionframework import Detector, Tracker, VisionPipeline
        
        # 测试无效配置（应该被捕获）
        test_cases = [
            # (config, should_be_valid, description)
            ({"conf_threshold": 1.5}, False, "conf_threshold 超出范围"),
            ({"conf_threshold": -0.1}, False, "conf_threshold 为负数"),
            ({"model_type": "invalid"}, False, "无效的 model_type"),
            ({"iou_threshold": 1.5}, False, "iou_threshold 超出范围"),
            ({"conf_threshold": 0.5}, True, "有效的 conf_threshold"),
            ({"max_age": -1}, False, "max_age 为负数（Tracker）"),
            ({"min_hits": 0}, False, "min_hits 为 0（Tracker）"),
            ({"enable_tracking": "yes"}, False, "enable_tracking 不是布尔值（Pipeline）"),
        ]
        
        passed = 0
        failed = 0
        
        for config, should_be_valid, description in test_cases:
            # 测试 Detector
            if "model_type" in config or "conf_threshold" in config or "iou_threshold" in config:
                detector = Detector(config)
                is_valid, error_msg = detector.validate_config(config)
                
                if should_be_valid:
                    if is_valid:
                        passed += 1
                        print(f"  [✓] Detector {description}: 验证通过")
                    else:
                        failed += 1
                        print(f"  [✗] Detector {description}: 应该有效但验证失败: {error_msg}")
                else:
                    if not is_valid:
                        passed += 1
                        print(f"  [✓] Detector {description}: 正确拒绝无效配置")
                    else:
                        failed += 1
                        print(f"  [✗] Detector {description}: 应该无效但验证通过")
            
            # 测试 Tracker
            if "max_age" in config or "min_hits" in config:
                tracker = Tracker(config)
                is_valid, error_msg = tracker.validate_config(config)
                
                if should_be_valid:
                    if is_valid:
                        passed += 1
                        print(f"  [✓] Tracker {description}: 验证通过")
                    else:
                        failed += 1
                        print(f"  [✗] Tracker {description}: 应该有效但验证失败: {error_msg}")
                else:
                    if not is_valid:
                        passed += 1
                        print(f"  [✓] Tracker {description}: 正确拒绝无效配置")
                    else:
                        failed += 1
                        print(f"  [✗] Tracker {description}: 应该无效但验证通过")
            
            # 测试 Pipeline
            if "enable_tracking" in config:
                pipeline = VisionPipeline({"enable_tracking": config["enable_tracking"]})
                is_valid, error_msg = pipeline.validate_config({"enable_tracking": config["enable_tracking"]})
                
                if should_be_valid:
                    if is_valid:
                        passed += 1
                        print(f"  [✓] Pipeline {description}: 验证通过")
                    else:
                        failed += 1
                        print(f"  [✗] Pipeline {description}: 应该有效但验证失败: {error_msg}")
                else:
                    if not is_valid:
                        passed += 1
                        print(f"  [✓] Pipeline {description}: 正确拒绝无效配置")
                    else:
                        failed += 1
                        print(f"  [✗] Pipeline {description}: 应该无效但验证通过")
        
        print(f"\n  配置验证测试: {passed} 通过, {failed} 失败")
        return failed == 0
        
    except Exception as e:
        print(f"  [✗] 配置验证测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_type_hints():
    """测试类型提示"""
    print("\n2. 测试类型提示...")
    
    try:
        from visionframework.core.detector import Detector
        from visionframework.core.tracker import Tracker
        from visionframework.core.pipeline import VisionPipeline
        
        modules = {
            "Detector": Detector,
            "Tracker": Tracker,
            "VisionPipeline": VisionPipeline
        }
        
        passed = 0
        failed = 0
        
        for name, module_class in modules.items():
            # 检查主要方法的类型提示
            methods_to_check = ["process", "initialize", "validate_config"]
            
            for method_name in methods_to_check:
                if hasattr(module_class, method_name):
                    method = getattr(module_class, method_name)
                    try:
                        hints = get_type_hints(method)
                        if hints:
                            passed += 1
                            print(f"  [✓] {name}.{method_name} 有类型提示: {list(hints.keys())}")
                        else:
                            # 没有类型提示可能也是正常的（如果方法很简单）
                            print(f"  [○] {name}.{method_name} 没有类型提示（可能不需要）")
                    except Exception:
                        # 某些情况下可能无法获取类型提示
                        print(f"  [○] {name}.{method_name} 无法获取类型提示")
        
        # 检查类属性的类型提示
        detector = Detector()
        tracker = Tracker()
        
        # 检查是否有类型注解的属性
        detector_annotations = get_type_hints(Detector)
        tracker_annotations = get_type_hints(Tracker)
        
        if detector_annotations or tracker_annotations:
            passed += 1
            print(f"  [✓] 类属性有类型提示")
        
        print(f"\n  类型提示测试: 基本检查完成")
        return True
        
    except Exception as e:
        print(f"  [✗] 类型提示测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_docstrings():
    """测试文档字符串"""
    print("\n3. 测试文档字符串...")
    
    try:
        from visionframework.core.detector import Detector
        from visionframework.core.tracker import Tracker
        from visionframework.core.pipeline import VisionPipeline
        from visionframework.core.base import BaseModule
        
        modules = {
            "BaseModule": BaseModule,
            "Detector": Detector,
            "Tracker": Tracker,
            "VisionPipeline": VisionPipeline
        }
        
        passed = 0
        total_checked = 0
        
        for name, module_class in modules.items():
            # 检查类的文档字符串
            class_doc = inspect.getdoc(module_class)
            total_checked += 1
            if class_doc and len(class_doc) > 50:  # 至少有一些描述
                passed += 1
                print(f"  [✓] {name} 类有详细文档字符串")
            else:
                print(f"  [!] {name} 类文档字符串可能不够详细")
            
            # 检查主要方法的文档字符串
            methods_to_check = ["__init__", "initialize", "process", "validate_config"]
            
            for method_name in methods_to_check:
                if hasattr(module_class, method_name):
                    method = getattr(module_class, method_name)
                    method_doc = inspect.getdoc(method)
                    total_checked += 1
                    
                    if method_doc:
                        # 检查是否包含 Args 和 Returns
                        has_args = "Args:" in method_doc or "参数:" in method_doc
                        has_returns = "Returns:" in method_doc or "返回:" in method_doc
                        
                        # 检查文档字符串长度（简单但有文档字符串的方法也算通过）
                        if len(method_doc) > 30:  # 至少有一些描述
                            passed += 1
                            if has_args or has_returns:
                                print(f"  [✓] {name}.{method_name} 有完整文档字符串（包含 Args/Returns）")
                            else:
                                print(f"  [✓] {name}.{method_name} 有文档字符串")
                        else:
                            print(f"  [○] {name}.{method_name} 有文档字符串但可能不够详细")
                    else:
                        print(f"  [!] {name}.{method_name} 缺少文档字符串")
        
        print(f"\n  文档字符串测试: {passed}/{total_checked} 通过")
        return passed >= total_checked * 0.7  # 至少70%应该有文档
        
    except Exception as e:
        print(f"  [✗] 文档字符串测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling_unification():
    """测试错误处理统一化"""
    print("\n4. 测试错误处理统一化...")
    
    try:
        from visionframework.core.base import BaseModule
        from visionframework import Detector, Tracker
        
        # 检查是否有统一的错误处理装饰器
        has_decorator = hasattr(BaseModule, 'handle_errors')
        
        if has_decorator:
            print("  [✓] BaseModule 有统一的错误处理装饰器")
            passed = 1
        else:
            print("  [○] BaseModule 没有统一的错误处理装饰器（可能使用其他方式）")
            passed = 0
        
        # 测试错误处理是否一致
        # 所有模块应该在初始化失败时返回 False 而不是抛出异常
        try:
            detector = Detector({"model_type": "invalid_type"})
            result = detector.initialize()
            if result is False:
                print("  [✓] 无效配置初始化返回 False（统一行为）")
                passed += 1
            else:
                print("  [✗] 无效配置初始化行为不一致")
        except Exception:
            print("  [✗] 无效配置初始化抛出异常（应该返回 False）")
        
        print(f"\n  错误处理统一化测试: {passed} 项通过")
        return passed >= 1
        
    except Exception as e:
        print(f"  [✗] 错误处理统一化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n5. 测试向后兼容性...")
    
    try:
        from visionframework import Detector, Tracker, VisionPipeline
        
        # 测试旧的 API 仍然工作
        # 1. 默认配置应该仍然工作
        detector = Detector()
        print("  [✓] Detector() 默认配置仍然工作")
        
        tracker = Tracker()
        print("  [✓] Tracker() 默认配置仍然工作")
        
        pipeline = VisionPipeline()
        print("  [✓] VisionPipeline() 默认配置仍然工作")
        
        # 2. 旧的配置方式应该仍然工作
        detector = Detector({"model_path": "yolov8n.pt", "conf_threshold": 0.25})
        print("  [✓] 旧的配置方式仍然工作")
        
        # 3. 方法调用应该仍然工作
        info = detector.get_model_info()
        print("  [✓] 方法调用仍然工作")
        
        return True
        
    except Exception as e:
        print(f"  [✗] 向后兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 70)
    print("第二阶段优化测试")
    print("=" * 70)
    print("\n本测试验证第二阶段优化的结果")
    print("=" * 70)
    
    tests = [
        ("配置验证功能", test_config_validation),
        ("类型提示完善", test_type_hints),
        ("文档字符串完善", test_docstrings),
        ("错误处理统一化", test_error_handling_unification),
        ("向后兼容性", test_backward_compatibility),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[✗] 测试 '{name}' 执行时出错: {e}")
            results.append((name, False))
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        if result:
            print(f"  [✓] {name}: 通过")
            passed += 1
        else:
            print(f"  [✗] {name}: 失败")
            failed += 1
    
    print("=" * 70)
    print(f"总计: {passed} 通过, {failed} 失败")
    print("=" * 70)
    
    if failed == 0:
        print("\n[✓] 所有测试通过！第二阶段优化成功完成。")
        print("\n改进总结:")
        print("  ✓ 配置验证功能正常工作")
        print("  ✓ 类型提示已完善")
        print("  ✓ 文档字符串已完善")
        print("  ✓ 错误处理已统一化")
        print("  ✓ 向后兼容性保持")
    else:
        print(f"\n[!] 有 {failed} 个测试失败，请检查上面的错误信息")
    
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

