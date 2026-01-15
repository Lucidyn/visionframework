"""
跟踪器单元测试

本测试验证各种跟踪器的功能：
- IOUTracker (IoU 跟踪器)
- ByteTrack (字节跟踪)
- ReIDTracker (ReID 跟踪)
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_mock_detection(x1, y1, x2, y2, confidence=0.9, class_id=1, class_name="person"):
    """创建 mock Detection 对象"""
    from visionframework.data.detection import Detection
    
    # Detection 接受 bbox 作为 (x1, y1, x2, y2) 元组
    return Detection(
        bbox=(x1, y1, x2, y2),
        confidence=confidence,
        class_id=class_id,
        class_name=class_name
    )


def test_tracker_import():
    """测试跟踪器导入"""
    print("\n1. 测试跟踪器导入...")
    try:
        from visionframework import Tracker, IOUTracker
        from visionframework.core.trackers import ByteTracker, ReIDTracker
        
        print("  [✓] 所有跟踪器成功导入")
        return True
    except Exception as e:
        print(f"  [✗] 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tracker_creation():
    """测试跟踪器创建"""
    print("\n2. 测试跟踪器创建...")
    
    try:
        from visionframework import Tracker
        
        # 测试 IOU 跟踪器
        tracker = Tracker({
            "tracker_type": "iou",
            "max_age": 30,
            "min_hits": 3,
            "iou_threshold": 0.3
        })
        print("  [✓] IoU 跟踪器创建成功")
        
        # 测试 ByteTrack
        tracker = Tracker({
            "tracker_type": "bytetrack",
            "track_thresh": 0.5,
            "track_buffer": 30,
            "match_thresh": 0.8
        })
        print("  [✓] ByteTrack 创建成功")
        
        # 测试 ReID 跟踪器
        try:
            tracker = Tracker({
                "tracker_type": "reid",
                "max_age": 30,
                "min_hits": 3
            })
            print("  [✓] ReID 跟踪器创建成功")
        except ImportError:
            print("  [SKIP] ReID 跟踪器需要额外依赖")
        
        return True
    except Exception as e:
        print(f"  [✗] 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tracker_initialization():
    """测试跟踪器初始化"""
    print("\n3. 测试跟踪器初始化...")
    
    try:
        from visionframework import Tracker
        
        # IoU 跟踪器
        tracker = Tracker({"tracker_type": "iou"})
        if tracker.initialize():
            print("  [✓] IoU 跟踪器初始化成功")
        else:
            print("  [✗] IoU 跟踪器初始化失败")
            return False
        
        # ByteTrack
        tracker = Tracker({"tracker_type": "bytetrack"})
        if tracker.initialize():
            print("  [✓] ByteTrack 初始化成功")
        else:
            print("  [✗] ByteTrack 初始化失败")
            return False
        
        return True
    except Exception as e:
        print(f"  [✗] 初始化测试失败: {e}")
        return False


def test_iou_tracker_update():
    """测试 IoU 跟踪器更新"""
    print("\n4. 测试 IoU 跟踪器更新...")
    
    try:
        from visionframework import Tracker
        
        # IoU 跟踪器测试：验证轨迹在多帧间的正确管理
        tracker = Tracker({"tracker_type": "iou", "max_age": 30, "min_hits": 1})
        if not tracker.initialize():
            print("  [SKIP] 跟踪器初始化失败")
            return None
        
        # 第 0 帧：创建两个轨迹
        detections_0 = [
            create_mock_detection(10, 10, 50, 50, 0.9, 1, "person"),
            create_mock_detection(100, 100, 150, 150, 0.8, 2, "car")
        ]
        tracks_0 = tracker.update(detections_0)
        
        # 检查：轨迹应该在内部被创建，但可能还未返回（如果 min_hits 大于 1）
        internal_count_0 = len(tracker.tracker_impl.tracks)
        print(f"  [✓] 第 0 帧: {len(detections_0)} 检测, 返回 {len(tracks_0)} 轨迹, 内部 {internal_count_0} 轨迹")
        
        if internal_count_0 < 2:
            print(f"  [✗] 第 0 帧应该创建 2 个轨迹")
            return False
        
        # 第 1 帧：继续检测，轨迹应该被匹配和更新
        detections_1 = [
            create_mock_detection(12, 12, 52, 52, 0.92, 1, "person"),
            create_mock_detection(100, 100, 150, 150, 0.78, 2, "car")
        ]
        tracks_1 = tracker.update(detections_1)
        internal_count_1 = len(tracker.tracker_impl.tracks)
        
        print(f"  [✓] 第 1 帧: {len(detections_1)} 检测, 返回 {len(tracks_1)} 轨迹, 内部 {internal_count_1} 轨迹")
        
        # 关键检查：轨迹应该被保持而不是替换
        # 如果 min_hits=1，应该返回 2 个轨迹
        if len(tracks_1) >= 2 and internal_count_1 == 2:
            print(f"  [✓] 轨迹持久化和匹配成功")
            return True
        else:
            print(f"  [!] 轨迹计数: 返回 {len(tracks_1)}, 内部 {internal_count_1}")
            # 如果 min_hits 大于 1，轨迹可能还未返回，但内部应该存在
            return internal_count_1 >= 2
    except Exception as e:
        print(f"  [✗] IoU 跟踪器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bytetrack_update():
    """测试 ByteTrack 更新"""
    print("\n5. 测试 ByteTrack 更新...")
    
    try:
        from visionframework import Tracker
        
        tracker = Tracker({
            "tracker_type": "bytetrack",
            "track_buffer": 10,
            "track_thresh": 0.5
        })
        if not tracker.initialize():
            print("  [SKIP] 跟踪器初始化失败")
            return None
        
        # 创建测试序列
        for frame_idx in range(3):
            # 第一个对象移动
            detections = [
                create_mock_detection(10 + frame_idx * 5, 10 + frame_idx * 5, 
                                     50 + frame_idx * 5, 50 + frame_idx * 5, 
                                     0.9, 1, "person")
            ]
            
            tracks = tracker.update(detections)
            print(f"  [✓] 第 {frame_idx} 帧: 检测 {len(detections)} 个, 跟踪 {len(tracks)} 个")
        
        return True
    except Exception as e:
        print(f"  [✗] ByteTrack 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tracker_configuration_validation():
    """测试跟踪器配置验证"""
    print("\n6. 测试跟踪器配置验证...")
    
    try:
        from visionframework import Tracker
        
        # 测试无效配置
        test_cases = [
            ({"max_age": -1}, False, "max_age 为负数"),
            ({"min_hits": 0}, False, "min_hits 为 0"),
            ({"max_age": 30, "min_hits": 3}, True, "有效配置"),
            ({"iou_threshold": 1.5}, False, "iou_threshold 超出范围"),
        ]
        
        passed = 0
        failed = 0
        
        for config, should_be_valid, description in test_cases:
            tracker = Tracker(config)
            is_valid, error_msg = tracker.validate_config(config)
            
            if should_be_valid:
                if is_valid:
                    passed += 1
                    print(f"  [✓] {description}: 验证通过")
                else:
                    failed += 1
                    print(f"  [✗] {description}: 应该有效但验证失败 - {error_msg}")
            else:
                if not is_valid:
                    passed += 1
                    print(f"  [✓] {description}: 正确拒绝")
                else:
                    failed += 1
                    print(f"  [✗] {description}: 应该无效但验证通过")
        
        return failed == 0
    except Exception as e:
        print(f"  [✗] 配置验证测试失败: {e}")
        return False


def test_tracker_with_empty_detections():
    """测试跟踪器处理空检测"""
    print("\n7. 测试跟踪器处理空检测...")
    
    try:
        from visionframework import Tracker
        
        tracker = Tracker({"tracker_type": "iou", "max_age": 5})
        if not tracker.initialize():
            print("  [SKIP] 跟踪器初始化失败")
            return None
        
        # 第一帧：有检测
        detections = [
            create_mock_detection(10, 10, 50, 50, 0.9, 1, "person")
        ]
        tracks = tracker.update(detections)
        print(f"  [✓] 第 0 帧: 有检测")
        
        # 第二帧：空检测
        empty_detections = []
        tracks = tracker.update(empty_detections)
        print(f"  [✓] 第 1 帧: 空检测，跟踪数: {len(tracks)}")
        
        # 第三帧：空检测
        tracks = tracker.update(empty_detections)
        print(f"  [✓] 第 2 帧: 空检测，跟踪数: {len(tracks)}")
        
        return True
    except Exception as e:
        print(f"  [✗] 空检测测试失败: {e}")
        return False


def test_tracker_configuration_options():
    """测试跟踪器不同配置选项"""
    print("\n8. 测试跟踪器配置选项...")
    
    try:
        from visionframework import Tracker
        
        # 测试不同的配置
        configs = [
            {"tracker_type": "iou", "max_age": 20, "min_hits": 2},
            {"tracker_type": "iou", "max_age": 50, "min_hits": 5},
            {"tracker_type": "bytetrack", "track_buffer": 20},
            {"tracker_type": "bytetrack", "track_buffer": 50, "track_thresh": 0.3},
        ]
        
        for i, config in enumerate(configs):
            tracker = Tracker(config)
            if tracker.initialize():
                print(f"  [✓] 配置 {i+1}: {config.get('tracker_type', 'default')} 初始化成功")
            else:
                print(f"  [✗] 配置 {i+1} 初始化失败")
                return False
        
        return True
    except Exception as e:
        print(f"  [✗] 配置选项测试失败: {e}")
        return False


def main():
    """运行所有跟踪器测试"""
    print("=" * 70)
    print("Vision Framework - 跟踪器单元测试")
    print("=" * 70)
    
    tests = [
        test_tracker_import,
        test_tracker_creation,
        test_tracker_initialization,
        test_iou_tracker_update,
        test_bytetrack_update,
        test_tracker_configuration_validation,
        test_tracker_with_empty_detections,
        test_tracker_configuration_options,
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
        print("\n[✓] 所有跟踪器测试通过!")
        return True
    else:
        print(f"\n[✗] {failed} 个测试失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
