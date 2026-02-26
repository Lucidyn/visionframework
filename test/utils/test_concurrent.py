"""
并发处理工具测试（Task、ThreadPoolProcessor、parallel_map）。
"""

import time
import pytest

from visionframework import Task, ThreadPoolProcessor, parallel_map


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

def test_task_creation():
    t = Task(task_id=1, data=42, func=lambda x: x * 2)
    assert t.task_id == 1
    assert t.data == 42
    assert t.result is None
    assert t.error is None


def test_task_execute():
    t = Task(task_id=1, data=10, func=lambda x: x + 5)
    result = t.execute()
    assert result == 15
    assert t.result == 15
    assert t.error is None


def test_task_execute_with_error():
    def bad_func(x):
        raise ValueError("intentional error")

    t = Task(task_id=1, data=0, func=bad_func)
    result = t.execute()
    assert result is None
    assert t.error is not None


def test_task_priority():
    t = Task(task_id=1, data=0, func=lambda x: x, priority=5)
    assert t.priority == 5


def test_task_max_retries():
    t = Task(task_id=1, data=0, func=lambda x: x, max_retries=3)
    assert t.max_retries == 3


def test_task_get_duration_before_execution():
    t = Task(task_id=1, data=0, func=lambda x: x)
    assert t.get_duration() is None


def test_task_get_duration_after_execution():
    t = Task(task_id=1, data=0, func=lambda x: time.sleep(0.01) or x)
    t.execute()
    dur = t.get_duration()
    assert dur is not None
    assert dur >= 0.0


# ---------------------------------------------------------------------------
# ThreadPoolProcessor
# ---------------------------------------------------------------------------

def test_thread_pool_processor_creation():
    proc = ThreadPoolProcessor(max_workers=2)
    assert isinstance(proc, ThreadPoolProcessor)


def test_thread_pool_processor_submit_and_get():
    proc = ThreadPoolProcessor(max_workers=2)
    task_id = proc.submit_task(5, lambda x: x * 2)
    time.sleep(0.5)
    status = proc.get_task_status(task_id)
    assert status is not None
    proc.stop()


def test_thread_pool_processor_submit_multiple():
    proc = ThreadPoolProcessor(max_workers=4)
    ids = [proc.submit_task(i, lambda x: x + 1) for i in range(5)]
    time.sleep(0.5)
    for tid in ids:
        status = proc.get_task_status(tid)
        assert status is not None
    proc.stop()


def test_thread_pool_processor_stop():
    proc = ThreadPoolProcessor(max_workers=2)
    proc.start()
    proc.stop()


def test_thread_pool_processor_process_batch():
    proc = ThreadPoolProcessor(max_workers=2)
    results = proc.process_batch([1, 2, 3, 4], lambda x: x ** 2)
    assert sorted(results) == [1, 4, 9, 16]
    proc.stop()


def test_thread_pool_processor_get_stats():
    proc = ThreadPoolProcessor(max_workers=2)
    proc.submit_task(1, lambda x: x)
    time.sleep(0.2)
    stats = proc.get_stats()
    assert isinstance(stats, dict)
    proc.stop()


# ---------------------------------------------------------------------------
# parallel_map（模块级辅助函数）
# ---------------------------------------------------------------------------

def test_parallel_map_basic():
    results = parallel_map([1, 2, 3], lambda x: x * 3)
    assert sorted(results) == [3, 6, 9]


def test_parallel_map_empty():
    results = parallel_map([], lambda x: x)
    assert results == []
