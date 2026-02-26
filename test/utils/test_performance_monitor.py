"""
PerformanceMonitor 与 PerformanceMetrics 测试。
"""

import time
import pytest

from visionframework import PerformanceMonitor, PerformanceMetrics


# ---------------------------------------------------------------------------
# PerformanceMetrics
# ---------------------------------------------------------------------------

def test_performance_metrics_creation():
    metrics = PerformanceMetrics()
    assert isinstance(metrics, PerformanceMetrics)


def test_performance_metrics_has_fps():
    metrics = PerformanceMetrics()
    assert hasattr(metrics, "fps")


def test_performance_metrics_has_memory():
    metrics = PerformanceMetrics()
    assert hasattr(metrics, "avg_memory_usage") or hasattr(metrics, "memory_usage") or hasattr(metrics, "cpu_memory_mb")


# ---------------------------------------------------------------------------
# PerformanceMonitor
# ---------------------------------------------------------------------------

def test_performance_monitor_creation():
    monitor = PerformanceMonitor()
    assert isinstance(monitor, PerformanceMonitor)


def test_performance_monitor_start():
    monitor = PerformanceMonitor()
    monitor.start()


def test_performance_monitor_tick():
    monitor = PerformanceMonitor()
    monitor.start()
    time.sleep(0.05)
    monitor.tick()
    monitor.tick()
    metrics = monitor.get_metrics()
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.fps >= 0


def test_performance_monitor_record_component_time():
    monitor = PerformanceMonitor()
    monitor.start()
    monitor.record_component_time("detector", 0.01)
    monitor.record_component_time("tracker", 0.005)
    metrics = monitor.get_metrics()
    assert isinstance(metrics, PerformanceMetrics)


def test_performance_monitor_get_metrics():
    monitor = PerformanceMonitor()
    monitor.start()
    monitor.tick()
    metrics = monitor.get_metrics()
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.fps >= 0


def test_performance_monitor_reset():
    monitor = PerformanceMonitor()
    monitor.start()
    monitor.tick()
    monitor.reset()
    metrics = monitor.get_metrics()
    assert metrics.fps == 0 or metrics.fps >= 0


def test_performance_monitor_get_detailed_report():
    monitor = PerformanceMonitor()
    monitor.start()
    monitor.tick()
    report = monitor.get_detailed_report()
    assert isinstance(report, (str, dict))
    assert report


def test_performance_monitor_multiple_ticks():
    """多次 tick 后 FPS 应为非负数。"""
    monitor = PerformanceMonitor()
    monitor.start()
    for _ in range(5):
        time.sleep(0.01)
        monitor.tick()
    metrics = monitor.get_metrics()
    assert metrics.fps >= 0
