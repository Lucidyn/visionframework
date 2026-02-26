"""
MemoryPool、MultiMemoryPool 及模块级辅助函数测试。
"""

import threading
import numpy as np
import pytest

from visionframework import (
    MemoryPool,
    MultiMemoryPool,
    create_memory_pool,
    acquire_memory,
    release_memory,
    get_memory_pool_status,
    optimize_memory_usage,
    clear_memory_pool,
    clear_all_memory_pools,
)


# ---------------------------------------------------------------------------
# MemoryPool
# ---------------------------------------------------------------------------

def test_memory_pool_creation():
    pool = MemoryPool(block_shape=(64, 64, 3), dtype=np.uint8, max_blocks=4)
    assert isinstance(pool, MemoryPool)


def test_memory_pool_acquire_returns_array():
    pool = MemoryPool(block_shape=(64, 64, 3), dtype=np.uint8, max_blocks=4)
    block = pool.acquire()
    assert isinstance(block, np.ndarray)
    assert block.shape == (64, 64, 3)
    assert block.dtype == np.uint8


def test_memory_pool_release_and_reuse():
    pool = MemoryPool(block_shape=(32, 32, 3), dtype=np.uint8, max_blocks=2)
    b1 = pool.acquire()
    pool.release(b1)
    b2 = pool.acquire()
    assert isinstance(b2, np.ndarray)


def test_memory_pool_multiple_blocks():
    pool = MemoryPool(block_shape=(16, 16, 3), dtype=np.uint8, max_blocks=4)
    blocks = [pool.acquire() for _ in range(4)]
    assert all(isinstance(b, np.ndarray) for b in blocks)
    for b in blocks:
        pool.release(b)


def test_memory_pool_stats():
    pool = MemoryPool(block_shape=(16, 16, 3), dtype=np.uint8, max_blocks=4)
    pool.acquire()
    stats = pool.stats
    assert isinstance(stats, dict)
    assert stats["acquires"] >= 1


def test_memory_pool_clear():
    pool = MemoryPool(block_shape=(8, 8, 3), dtype=np.uint8, max_blocks=4)
    b = pool.acquire()
    pool.release(b)
    pool.clear()


def test_memory_pool_thread_safety():
    pool = MemoryPool(block_shape=(8, 8, 3), dtype=np.uint8, max_blocks=8)
    errors = []

    def worker():
        try:
            for _ in range(5):
                b = pool.acquire()
                b[...] = 0
                pool.release(b)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"线程错误：{errors}"


def test_memory_pool_min_blocks_preallocation():
    pool = MemoryPool(block_shape=(8, 8, 3), dtype=np.uint8, max_blocks=4, min_blocks=2)
    b = pool.acquire()
    assert isinstance(b, np.ndarray)
    pool.release(b)


# ---------------------------------------------------------------------------
# MultiMemoryPool
# ---------------------------------------------------------------------------

def test_multi_memory_pool_creation():
    mpool = MultiMemoryPool()
    assert isinstance(mpool, MultiMemoryPool)


def test_multi_memory_pool_create_and_acquire():
    mpool = MultiMemoryPool()
    mpool.create_pool("frames", block_shape=(64, 64, 3), dtype=np.uint8, max_blocks=4)
    block = mpool.acquire("frames")
    assert isinstance(block, np.ndarray)
    mpool.release("frames", block)


def test_multi_memory_pool_get_status():
    mpool = MultiMemoryPool()
    mpool.create_pool("p1", block_shape=(16, 16, 3), dtype=np.uint8, max_blocks=2)
    status = mpool.get_status()
    assert isinstance(status, dict)


# ---------------------------------------------------------------------------
# 模块级辅助函数
# ---------------------------------------------------------------------------

def test_create_and_acquire_release():
    create_memory_pool("test_pool", block_shape=(32, 32, 3), dtype=np.uint8, max_blocks=4)
    block = acquire_memory("test_pool")
    assert isinstance(block, np.ndarray)
    release_memory("test_pool", block)
    clear_memory_pool("test_pool")


def test_get_memory_pool_status():
    create_memory_pool("status_pool", block_shape=(8, 8, 3), dtype=np.uint8, max_blocks=2)
    status = get_memory_pool_status()
    assert isinstance(status, dict)
    clear_memory_pool("status_pool")


def test_optimize_memory_usage():
    optimize_memory_usage()


def test_clear_all_memory_pools():
    create_memory_pool("pool_a", block_shape=(8, 8, 3), dtype=np.uint8, max_blocks=2)
    create_memory_pool("pool_b", block_shape=(8, 8, 3), dtype=np.uint8, max_blocks=2)
    clear_all_memory_pools()
