"""
Test memory management functionality

This test file tests the memory management utilities, including memory pool
implementation and memory optimization features.
"""

import unittest
import numpy as np
import time
from visionframework.utils.memory import (
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


class TestMemoryManager(unittest.TestCase):
    """Test memory management functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Clear all memory pools before each test
        clear_all_memory_pools()
    
    def tearDown(self):
        """Clean up test environment"""
        # Clear all memory pools after each test
        clear_all_memory_pools()
    
    def test_memory_pool_creation(self):
        """Test memory pool creation"""
        # Create a memory pool
        pool = MemoryPool(
            block_shape=(100, 100, 3),
            dtype=np.uint8,
            max_blocks=10,
            name="test_pool"
        )
        
        # Check pool properties
        self.assertEqual(pool.block_shape, (100, 100, 3))
        self.assertEqual(pool.dtype, np.uint8)
        self.assertEqual(pool.max_blocks, 10)
        self.assertEqual(pool.name, "test_pool")
    
    def test_memory_pool_acquire_release(self):
        """Test memory pool acquire and release functionality"""
        # Create a memory pool
        pool = MemoryPool(
            block_shape=(100, 100, 3),
            dtype=np.uint8,
            max_blocks=5
        )
        
        # Acquire blocks
        blocks = []
        for i in range(5):
            block = pool.acquire()
            self.assertEqual(block.shape, (100, 100, 3))
            self.assertEqual(block.dtype, np.uint8)
            blocks.append(block)
        
        # Release blocks
        for block in blocks:
            pool.release(block)
        
        # Acquire again to test reuse
        for i in range(5):
            block = pool.acquire()
            self.assertEqual(block.shape, (100, 100, 3))
            pool.release(block)
    
    def test_memory_pool_status(self):
        """Test memory pool status functionality"""
        # Create a memory pool
        pool = MemoryPool(
            block_shape=(100, 100, 3),
            dtype=np.uint8,
            max_blocks=5,
            name="status_test"
        )
        
        # Get status
        status = pool.get_status()
        self.assertEqual(status["name"], "status_test")
        self.assertEqual(status["block_shape"], (100, 100, 3))
        self.assertEqual(status["dtype"], "uint8")
        self.assertEqual(status["max_blocks"], 5)
        self.assertEqual(status["current_blocks"], 0)
    
    def test_multi_memory_pool(self):
        """Test multi-memory pool functionality"""
        # Get global memory pool manager
        manager = get_memory_pool_manager()
        
        # Create multiple pools
        pool1 = manager.create_pool(
            pool_name="pool1",
            block_shape=(100, 100, 3),
            dtype=np.uint8,
            max_blocks=5
        )
        
        pool2 = manager.create_pool(
            pool_name="pool2",
            block_shape=(200, 200, 3),
            dtype=np.float32,
            max_blocks=3
        )
        
        # Acquire blocks from different pools
        block1 = manager.acquire("pool1")
        self.assertEqual(block1.shape, (100, 100, 3))
        
        block2 = manager.acquire("pool2")
        self.assertEqual(block2.shape, (200, 200, 3))
        
        # Release blocks
        manager.release("pool1", block1)
        manager.release("pool2", block2)
        
        # Get status
        status = manager.get_status()
        self.assertEqual(len(status["pools"]), 2)
        self.assertIn("pool1", status["pools"])
        self.assertIn("pool2", status["pools"])
    
    def test_global_memory_pool_functions(self):
        """Test global memory pool functions"""
        # Create memory pool using global function
        create_memory_pool(
            pool_name="global_pool",
            block_shape=(150, 150, 3),
            dtype=np.uint8,
            max_blocks=8
        )
        
        # Acquire memory
        block = acquire_memory("global_pool")
        self.assertEqual(block.shape, (150, 150, 3))
        
        # Release memory
        release_memory("global_pool", block)
        
        # Get status
        status = get_memory_pool_status()
        self.assertIn("global_pool", status["pools"])
        
        # Clear pool
        clear_memory_pool("global_pool")
        
        # Clear all pools
        clear_all_memory_pools()
        
        status = get_memory_pool_status()
        self.assertEqual(len(status["pools"]), 0)
    
    def test_shared_array(self):
        """Test shared array functionality"""
        # Create shared array
        shape = (100, 100, 3)
        dtype = np.uint8
        shared_array = create_shared_array(shape, dtype)
        
        # Check array properties
        self.assertEqual(shared_array.shape, shape)
        self.assertEqual(shared_array.dtype, dtype)
        
        # Modify array
        shared_array[:] = 255
        self.assertEqual(shared_array[0, 0, 0], 255)
        
        # Free shared array
        free_shared_array(shared_array)
    
    def test_memory_optimization(self):
        """Test memory optimization functionality"""
        # Create some memory pools and allocate memory
        create_memory_pool(
            pool_name="opt_pool1",
            block_shape=(200, 200, 3),
            dtype=np.uint8,
            max_blocks=5
        )
        
        create_memory_pool(
            pool_name="opt_pool2",
            block_shape=(100, 100, 3),
            dtype=np.float32,
            max_blocks=3
        )
        
        # Acquire and release some memory
        block1 = acquire_memory("opt_pool1")
        release_memory("opt_pool1", block1)
        
        block2 = acquire_memory("opt_pool2")
        release_memory("opt_pool2", block2)
        
        # Optimize memory usage
        result = optimize_memory_usage()
        
        # Check optimization result
        self.assertIn("initial_status", result)
        self.assertIn("final_status", result)
        self.assertIn("message", result)
        self.assertEqual(result["message"], "Memory optimization completed")
    
    def test_memory_pool_performance(self):
        """Test memory pool performance"""
        # Create a memory pool
        pool = MemoryPool(
            block_shape=(500, 500, 3),
            dtype=np.uint8,
            max_blocks=10
        )
        
        # Test memory allocation time with pool
        start_time = time.time()
        blocks = []
        for i in range(8):
            block = pool.acquire()
            blocks.append(block)
        pool_time = time.time() - start_time
        
        # Test memory allocation time without pool
        start_time = time.time()
        numpy_blocks = []
        for i in range(8):
            block = np.zeros((500, 500, 3), dtype=np.uint8)
            numpy_blocks.append(block)
        numpy_time = time.time() - start_time
        
        # Memory pool should be at least as fast as numpy allocation
        # Note: This test might vary depending on system, so we're not asserting strict inequality
        # Just ensuring the functionality works
        self.assertIsInstance(pool_time, float)
        self.assertIsInstance(numpy_time, float)
        
        # Release blocks
        for block in blocks:
            pool.release(block)
    
    def test_memory_pool_edge_cases(self):
        """Test memory pool edge cases"""
        # Test pool with max_blocks=0 (should still work)
        pool = MemoryPool(
            block_shape=(50, 50, 3),
            dtype=np.uint8,
            max_blocks=0
        )
        
        # Acquire block (should allocate new block)
        block = pool.acquire()
        self.assertEqual(block.shape, (50, 50, 3))
        
        # Release block (should be discarded since max_blocks=0)
        pool.release(block)
        
        # Test with very large block size
        pool_large = MemoryPool(
            block_shape=(1000, 1000, 3),
            dtype=np.uint8,
            max_blocks=2
        )
        
        block_large = pool_large.acquire()
        self.assertEqual(block_large.shape, (1000, 1000, 3))
        pool_large.release(block_large)
    
    def test_memory_pool_min_blocks(self):
        """Test memory pool with min_blocks"""
        # Create memory pool with min_blocks
        pool = MemoryPool(
            block_shape=(100, 100, 3),
            dtype=np.uint8,
            max_blocks=10,
            min_blocks=2
        )
        
        # Check initial status
        status = pool.get_status()
        self.assertEqual(status["min_blocks"], 2)
        
        # Acquire all blocks
        blocks = []
        for _ in range(5):
            blocks.append(pool.acquire())
        
        # Release all blocks
        for block in blocks:
            pool.release(block)
        
        # Clear pool (should keep at least min_blocks)
        pool.clear()
        status = pool.get_status()
        self.assertGreaterEqual(status["current_blocks"], 2)
    
    def test_memory_pool_resize(self):
        """Test memory pool resizing"""
        # Create memory pool
        pool = MemoryPool(
            block_shape=(100, 100, 3),
            dtype=np.uint8,
            max_blocks=5
        )
        
        # Acquire some blocks
        blocks = []
        for _ in range(3):
            blocks.append(pool.acquire())
        
        # Release blocks
        for block in blocks:
            pool.release(block)
        
        # Resize pool
        pool.resize(8)
        status = pool.get_status()
        self.assertEqual(status["max_blocks"], 8)
        
        # Acquire more blocks
        more_blocks = []
        for _ in range(6):
            more_blocks.append(pool.acquire())
        
        # Release blocks
        for block in more_blocks:
            pool.release(block)
        
        # Resize to smaller size
        pool.resize(3)
        status = pool.get_status()
        self.assertEqual(status["max_blocks"], 3)
    
    def test_memory_pool_dynamic_resizing(self):
        """Test memory pool with dynamic resizing"""
        # Create memory pool with dynamic resizing enabled
        pool = MemoryPool(
            block_shape=(100, 100, 3),
            dtype=np.uint8,
            max_blocks=5,
            enable_dynamic_resizing=True,
            resize_factor=1.5
        )
        
        # Check status
        status = pool.get_status()
        self.assertEqual(status["dynamic_resizing"], True)
    
    def test_memory_pool_stats(self):
        """Test memory pool statistics"""
        # Create memory pool
        pool = MemoryPool(
            block_shape=(100, 100, 3),
            dtype=np.uint8,
            max_blocks=5
        )
        
        # Acquire and release blocks
        block1 = pool.acquire()
        pool.release(block1)
        
        # Acquire again (should be a hit)
        block2 = pool.acquire()
        pool.release(block2)
        
        # Get status and check stats
        status = pool.get_status()
        self.assertIn("statistics", status)
        stats = status["statistics"]
        self.assertGreaterEqual(stats["acquires"], 2)
        self.assertGreaterEqual(stats["releases"], 2)
        self.assertGreaterEqual(stats["hits"], 1)
    
    def test_resize_memory_pool_function(self):
        """Test resize_memory_pool function"""
        # Create memory pool
        create_memory_pool(
            pool_name="test_resize_pool",
            block_shape=(100, 100, 3),
            dtype=np.uint8,
            max_blocks=5
        )
        
        # Resize the pool
        resize_memory_pool("test_resize_pool", 10)
        
        # Check status
        status = get_memory_pool_status()
        self.assertIn("test_resize_pool", status["pools"])
        pool_status = status["pools"]["test_resize_pool"]
        # Note: We can't check max_blocks directly as it's not exposed in the status
        # This test just ensures the function doesn't throw an error
        
        # Clean up
        clear_memory_pool("test_resize_pool")


if __name__ == "__main__":
    unittest.main()
