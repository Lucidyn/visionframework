"""
Memory utilities module

This module provides memory management utilities, including memory pool
implementation to reduce memory allocation overhead and improve performance.
"""

from .memory_manager import (
    MemoryPool,
    MultiMemoryPool,
    get_memory_pool_manager,
    create_memory_pool,
    acquire_memory,
    release_memory,
    clear_memory_pool,
    clear_all_memory_pools,
    resize_memory_pool,
    get_memory_pool_status,
    create_shared_array,
    free_shared_array,
    optimize_memory_usage
)

__all__ = [
    "MemoryPool",
    "MultiMemoryPool",
    "get_memory_pool_manager",
    "create_memory_pool",
    "acquire_memory",
    "release_memory",
    "clear_memory_pool",
    "clear_all_memory_pools",
    "resize_memory_pool",
    "get_memory_pool_status",
    "create_shared_array",
    "free_shared_array",
    "optimize_memory_usage"
]
