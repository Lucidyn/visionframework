"""
Memory management utilities

This module provides memory management utilities, including memory pool
implementation to reduce memory allocation overhead and improve performance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import threading
import logging

logger = logging.getLogger(__name__)


class MemoryPool:
    """
    Memory pool for efficient memory allocation and reuse
    
    This class implements a memory pool that pre-allocates memory blocks
    and reuses them to reduce the overhead of frequent memory allocation
    and deallocation operations.
    
    Example:
        ```python
        # Create a memory pool for numpy arrays
        pool = MemoryPool(
            block_shape=(640, 640, 3),
            dtype=np.uint8,
            max_blocks=10
        )
        
        # Get a memory block from the pool
        block = pool.acquire()
        
        # Use the block
        block[...] = 0  # Clear the block
        # Process the block...
        
        # Return the block to the pool
        pool.release(block)
        ```
    """
    
    def __init__(self,
                 block_shape: Tuple[int, ...],
                 dtype: np.dtype = np.uint8,
                 max_blocks: int = 10,
                 name: Optional[str] = None):
        """
        Initialize memory pool
        
        Args:
            block_shape: Shape of each memory block
            dtype: Data type of memory blocks
            max_blocks: Maximum number of blocks to keep in the pool
            name: Optional name for the pool (for logging)
        """
        self.block_shape = block_shape
        self.dtype = dtype
        self.max_blocks = max_blocks
        self.name = name or f"MemoryPool_{hex(id(self))[:8]}"
        
        # Create a queue to store memory blocks
        self._blocks: deque = deque(maxlen=max_blocks)
        self._lock = threading.RLock()  # Thread-safe
        
        # Calculate block size for logging
        self.block_size = np.prod(block_shape) * np.dtype(dtype).itemsize
        self.total_size = 0
        
        logger.info(f"Created {self.name} with block shape {block_shape}, dtype {dtype},")
        logger.info(f"  block size: {self.block_size / 1024:.2f} KB, max blocks: {max_blocks}")
    
    def acquire(self) -> np.ndarray:
        """
        Acquire a memory block from the pool
        
        If the pool is empty, a new block is allocated. Otherwise, a reused
        block is returned.
        
        Returns:
            np.ndarray: Memory block from the pool
        """
        with self._lock:
            if self._blocks:
                # Reuse an existing block
                block = self._blocks.popleft()
                logger.debug(f"{self.name}: Acquired reused block, remaining: {len(self._blocks)}")
                return block
            else:
                # Allocate a new block
                block = np.zeros(self.block_shape, dtype=self.dtype)
                self.total_size += self.block_size
                logger.debug(f"{self.name}: Allocated new block, total allocated: {self.total_size / 1024:.2f} KB")
                return block
    
    def release(self, block: np.ndarray) -> None:
        """
        Release a memory block back to the pool
        
        Args:
            block: Memory block to return to the pool
        """
        with self._lock:
            if len(self._blocks) < self.max_blocks:
                # Reset the block to avoid data leakage
                block.fill(0)
                self._blocks.append(block)
                logger.debug(f"{self.name}: Released block, pool size: {len(self._blocks)}")
            else:
                # Pool is full, let the block be garbage collected
                logger.debug(f"{self.name}: Pool full, discarding block")
    
    def clear(self) -> None:
        """
        Clear all blocks in the pool
        """
        with self._lock:
            blocks_cleared = len(self._blocks)
            self._blocks.clear()
            logger.info(f"{self.name}: Cleared {blocks_cleared} blocks")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get pool status
        
        Returns:
            Dict[str, Any]: Pool status information
        """
        with self._lock:
            return {
                "name": self.name,
                "block_shape": self.block_shape,
                "dtype": str(self.dtype),
                "block_size_kb": self.block_size / 1024,
                "max_blocks": self.max_blocks,
                "current_blocks": len(self._blocks),
                "total_allocated_kb": self.total_size / 1024
            }


class MultiMemoryPool:
    """
    Multi-type memory pool manager
    
    This class manages multiple memory pools for different memory types,
    allowing efficient memory allocation and reuse for various use cases.
    """
    
    def __init__(self):
        """
        Initialize multi-memory pool manager
        """
        self._pools: Dict[str, MemoryPool] = {}
        self._lock = threading.RLock()
        logger.info("Created MultiMemoryPool manager")
    
    def create_pool(self,
                    pool_name: str,
                    block_shape: Tuple[int, ...],
                    dtype: np.dtype = np.uint8,
                    max_blocks: int = 10) -> MemoryPool:
        """
        Create a new memory pool
        
        Args:
            pool_name: Name of the pool
            block_shape: Shape of each memory block
            dtype: Data type of memory blocks
            max_blocks: Maximum number of blocks to keep in the pool
        
        Returns:
            MemoryPool: Created memory pool
        """
        with self._lock:
            if pool_name in self._pools:
                logger.warning(f"Pool '{pool_name}' already exists, returning existing pool")
                return self._pools[pool_name]
            
            pool = MemoryPool(
                block_shape=block_shape,
                dtype=dtype,
                max_blocks=max_blocks,
                name=pool_name
            )
            self._pools[pool_name] = pool
            logger.info(f"Created pool '{pool_name}'")
            return pool
    
    def get_pool(self, pool_name: str) -> Optional[MemoryPool]:
        """
        Get an existing memory pool
        
        Args:
            pool_name: Name of the pool
        
        Returns:
            Optional[MemoryPool]: Memory pool if exists, None otherwise
        """
        with self._lock:
            return self._pools.get(pool_name)
    
    def acquire(self, pool_name: str) -> Optional[np.ndarray]:
        """
        Acquire a memory block from a specific pool
        
        Args:
            pool_name: Name of the pool
        
        Returns:
            Optional[np.ndarray]: Memory block if pool exists, None otherwise
        """
        pool = self.get_pool(pool_name)
        if pool:
            return pool.acquire()
        else:
            logger.error(f"Pool '{pool_name}' does not exist")
            return None
    
    def release(self, pool_name: str, block: np.ndarray) -> None:
        """
        Release a memory block back to a specific pool
        
        Args:
            pool_name: Name of the pool
            block: Memory block to return
        """
        pool = self.get_pool(pool_name)
        if pool:
            pool.release(block)
        else:
            logger.error(f"Pool '{pool_name}' does not exist")
    
    def clear_pool(self, pool_name: str) -> None:
        """
        Clear a specific memory pool
        
        Args:
            pool_name: Name of the pool
        """
        pool = self.get_pool(pool_name)
        if pool:
            pool.clear()
        else:
            logger.error(f"Pool '{pool_name}' does not exist")
    
    def clear_all(self) -> None:
        """
        Clear all memory pools
        """
        with self._lock:
            for pool_name, pool in self._pools.items():
                pool.clear()
            logger.info("Cleared all memory pools")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of all memory pools
        
        Returns:
            Dict[str, Any]: Status information for all pools
        """
        with self._lock:
            status = {
                "total_pools": len(self._pools),
                "pools": {}
            }
            for pool_name, pool in self._pools.items():
                status["pools"][pool_name] = pool.get_status()
            return status


# Global memory pool manager instance
_memory_pool_manager: Optional[MultiMemoryPool] = None
_memory_pool_lock = threading.Lock()


def get_memory_pool_manager() -> MultiMemoryPool:
    """
    Get global memory pool manager instance
    
    Returns:
        MultiMemoryPool: Global memory pool manager instance
    """
    global _memory_pool_manager
    with _memory_pool_lock:
        if _memory_pool_manager is None:
            _memory_pool_manager = MultiMemoryPool()
        return _memory_pool_manager


def create_memory_pool(
    pool_name: str,
    block_shape: Tuple[int, ...],
    dtype: np.dtype = np.uint8,
    max_blocks: int = 10) -> MemoryPool:
    """
    Create a memory pool using the global manager
    
    Args:
        pool_name: Name of the pool
        block_shape: Shape of each memory block
        dtype: Data type of memory blocks
        max_blocks: Maximum number of blocks to keep in the pool
    
    Returns:
        MemoryPool: Created memory pool
    """
    manager = get_memory_pool_manager()
    return manager.create_pool(
        pool_name=pool_name,
        block_shape=block_shape,
        dtype=dtype,
        max_blocks=max_blocks
    )


def acquire_memory(pool_name: str) -> Optional[np.ndarray]:
    """
    Acquire memory from a specific pool
    
    Args:
        pool_name: Name of the pool
    
    Returns:
        Optional[np.ndarray]: Memory block if pool exists, None otherwise
    """
    manager = get_memory_pool_manager()
    return manager.acquire(pool_name)


def release_memory(pool_name: str, block: np.ndarray) -> None:
    """
    Release memory back to a specific pool
    
    Args:
        pool_name: Name of the pool
        block: Memory block to return
    """
    manager = get_memory_pool_manager()
    manager.release(pool_name, block)


def clear_memory_pool(pool_name: str) -> None:
    """
    Clear a specific memory pool
    
    Args:
        pool_name: Name of the pool
    """
    manager = get_memory_pool_manager()
    manager.clear_pool(pool_name)


def clear_all_memory_pools() -> None:
    """
    Clear all memory pools
    """
    manager = get_memory_pool_manager()
    manager.clear_all()


def get_memory_pool_status() -> Dict[str, Any]:
    """
    Get status of all memory pools
    
    Returns:
        Dict[str, Any]: Status information for all pools
    """
    manager = get_memory_pool_manager()
    return manager.get_status()


# Memory utility functions
def create_shared_array(shape: Tuple[int, ...], dtype: np.dtype = np.uint8) -> np.ndarray:
    """
    Create a shared memory array that can be used across processes
    
    Args:
        shape: Shape of the array
        dtype: Data type of the array
    
    Returns:
        np.ndarray: Shared memory array
    """
    import multiprocessing as mp
    
    # Calculate size
    size = np.prod(shape) * np.dtype(dtype).itemsize
    
    # Create shared memory
    shared_mem = mp.shared_memory.SharedMemory(create=True, size=size)
    
    # Create numpy array backed by shared memory
    array = np.ndarray(shape, dtype=dtype, buffer=shared_mem.buf)
    
    # Store shared memory handle in array's metadata
    array.__shared_memory__ = shared_mem
    
    return array


def free_shared_array(array: np.ndarray) -> None:
    """
    Free a shared memory array
    
    Args:
        array: Shared memory array to free
    """
    if hasattr(array, '__shared_memory__'):
        shared_mem = array.__shared_memory__
        shared_mem.close()
        shared_mem.unlink()
    else:
        logger.warning("Array is not a shared memory array")


def optimize_memory_usage() -> Dict[str, Any]:
    """
    Optimize memory usage by clearing unused memory
    
    Returns:
        Dict[str, Any]: Optimization results
    """
    import gc
    
    # Get initial memory usage
    initial_status = get_memory_pool_status()
    
    # Clear unused memory pools
    clear_all_memory_pools()
    
    # Run garbage collection
    gc.collect()
    
    # Get final memory usage
    final_status = get_memory_pool_status()
    
    return {
        "initial_status": initial_status,
        "final_status": final_status,
        "message": "Memory optimization completed"
    }
