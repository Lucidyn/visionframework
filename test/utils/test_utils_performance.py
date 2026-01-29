"""
测试性能监控工具类
"""

import time
import pytest
from visionframework.utils.monitoring.performance import PerformanceMonitor, Timer


# 测试PerformanceMonitor类
def test_performance_monitor_initialization():
    """测试性能监控器初始化"""
    monitor = PerformanceMonitor(window_size=60)
    assert monitor.window_size == 60
    assert len(monitor.frame_times) == 0
    assert "detection" in monitor.component_times
    assert "tracking" in monitor.component_times
    assert "visualization" in monitor.component_times
    
    monitor = PerformanceMonitor(window_size=30)
    assert monitor.window_size == 30


def test_performance_monitor_start_tick():
    """测试性能监控器的start和tick方法"""
    monitor = PerformanceMonitor()
    
    # 测试start方法
    monitor.start()
    assert monitor.start_time is not None
    assert monitor.last_frame_time is not None
    
    # 测试tick方法
    time.sleep(0.01)  # 等待10ms
    monitor.tick()
    assert len(monitor.frame_times) == 1
    
    time.sleep(0.01)
    monitor.tick()
    assert len(monitor.frame_times) == 2


def test_performance_monitor_record_component_time():
    """测试记录组件时间"""
    monitor = PerformanceMonitor()
    
    # 测试记录有效组件时间
    monitor.record_component_time("detection", 0.123)
    monitor.record_component_time("tracking", 0.045)
    monitor.record_component_time("visualization", 0.067)
    
    assert len(monitor.component_times["detection"]) == 1
    assert len(monitor.component_times["tracking"]) == 1
    assert len(monitor.component_times["visualization"]) == 1
    
    # 测试记录无效组件时间（应该被忽略）
    monitor.record_component_time("invalid_component", 0.999)
    assert len(monitor.component_times) == 6  # 不应该增加新的组件，默认有6个组件


def test_performance_monitor_get_metrics():
    """测试获取性能指标"""
    monitor = PerformanceMonitor()
    
    # 测试空指标
    metrics = monitor.get_metrics()
    assert metrics.fps == 0.0
    assert metrics.avg_fps == 0.0
    assert metrics.frame_count == 0
    assert metrics.total_time == 0.0
    assert metrics.avg_time_per_frame == 0.0
    
    # 测试有数据的指标
    monitor.start()
    for i in range(10):
        time.sleep(0.001)
        monitor.tick()
        monitor.record_component_time("detection", 0.01 * (i + 1))
        monitor.record_component_time("tracking", 0.005 * (i + 1))
    
    metrics = monitor.get_metrics()
    assert metrics.frame_count == 10
    assert metrics.fps > 0
    assert metrics.avg_fps > 0
    assert metrics.total_time > 0
    assert metrics.avg_time_per_frame > 0


def test_performance_monitor_get_detailed_report():
    """测试获取详细性能报告"""
    monitor = PerformanceMonitor()
    
    # 测试空报告
    report = monitor.get_detailed_report()
    assert "timestamp" in report
    assert "fps" in report
    assert "frame_times" in report
    assert "component_times_ms" in report
    assert "memory" in report
    assert "general" in report
    
    # 测试有数据的报告
    monitor.start()
    for i in range(10):
        time.sleep(0.001)
        monitor.tick()
        monitor.record_component_time("detection", 0.01 * (i + 1))
        monitor.record_component_time("tracking", 0.005 * (i + 1))
    
    report = monitor.get_detailed_report()
    assert report["general"]["frame_count"] == 10
    assert report["fps"]["current"] > 0
    assert report["fps"]["average"] > 0


def test_performance_monitor_reset():
    """测试重置性能监控器"""
    monitor = PerformanceMonitor()
    monitor.start()
    monitor.tick()
    monitor.record_component_time("detection", 0.123)
    
    # 重置前应该有数据
    assert len(monitor.frame_times) == 1
    assert len(monitor.component_times["detection"]) == 1
    
    # 测试重置
    monitor.reset()
    
    # 重置后应该清空数据
    assert len(monitor.frame_times) == 0
    assert len(monitor.component_times["detection"]) == 0
    assert monitor.start_time is None
    assert monitor.last_frame_time is None


# 测试Timer类
def test_timer_context_manager():
    """测试Timer上下文管理器"""
    with Timer("test_operation") as timer:
        time.sleep(0.01)
    
    assert timer.get_elapsed() > 0
    assert isinstance(timer.get_elapsed(), float)


def test_timer_manual():
    """测试手动使用Timer"""
    timer = Timer("test_manual")
    assert timer.get_elapsed() == 0.0
    
    timer.__enter__()
    time.sleep(0.01)
    timer.__exit__(None, None, None)
    
    elapsed = timer.get_elapsed()
    assert elapsed > 0
    assert isinstance(elapsed, float)


def test_timer_get_elapsed():
    """测试获取经过时间"""
    with Timer() as timer:
        time.sleep(0.02)
    
    elapsed = timer.get_elapsed()
    assert 0.01 < elapsed < 0.1  # 允许一定的时间误差


def test_timer_name():
    """测试Timer名称"""
    timer = Timer("custom_name")
    assert timer.name == "custom_name"
    
    timer2 = Timer()
    assert timer2.name == "Operation"  # 默认名称
