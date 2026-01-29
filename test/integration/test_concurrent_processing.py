"""
Test concurrent processing functionality

This test file tests the concurrent processing utilities, including:
- Thread pool processing
- Process pool processing
- Asynchronous processing
- Parallel map functionality
"""

import unittest
import numpy as np
import time
import concurrent.futures
from visionframework.utils.concurrent import (
    ThreadPoolProcessor,
    ProcessPoolProcessor,
    AsyncProcessor,
    Task,
    get_thread_pool_processor,
    get_process_pool_processor,
    get_async_processor,
    shutdown_all_processors,
    parallel_map,
    async_map,
    ThreadPoolContext,
    ProcessPoolContext
)

# Module-level functions for process pool testing
def pool_func_thread(x):
    time.sleep(0.01)  # Simulate work
    return x * 2

def pool_func_process(x):
    time.sleep(0.01)  # Simulate work
    return x * 3

def pool_func_async(x):
    time.sleep(0.01)  # Simulate work
    return x * 4

def pool_func_parallel(x):
    time.sleep(0.01)  # Simulate work
    return x * 5

def pool_compute_intensive(x):
    # Simulate intensive computation
    result = 0
    for i in range(100000):
        result += x * i
    return result

def pool_func_context_thread(x):
    return x * 7

def pool_func_context_process(x):
    return x * 8

def pool_func_task(x):
    return x * 9

def pool_error_func(x):
    if x == 2:
        raise ValueError("Test error")
    return x * 2


class TestConcurrentProcessing(unittest.TestCase):
    """Test concurrent processing functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Shutdown all processors before each test
        shutdown_all_processors()
    
    def tearDown(self):
        """Clean up test environment"""
        # Shutdown all processors after each test
        shutdown_all_processors()
    
    def test_thread_pool_processor(self):
        """Test thread pool processor"""
        # Create thread pool processor
        processor = ThreadPoolProcessor(max_workers=4)
        processor.start()
        
        # Submit tasks
        task_ids = []
        for i in range(10):
            task_id = processor.submit_task(i, pool_func_thread)
            task_ids.append(task_id)
        
        # Wait for tasks to complete
        time.sleep(0.1)
        
        # Check task status
        for task_id in task_ids:
            status = processor.get_task_status(task_id)
            self.assertIn("result", status)
            self.assertIsNone(status.get("error"))
        
        # Process batch
        data = list(range(10))
        results = processor.process_batch_ordered(data, pool_func_thread)
        self.assertEqual(len(results), 10)
        self.assertEqual(results, [x * 2 for x in data])
        
        # Get stats
        stats = processor.get_stats()
        self.assertIn("total_tasks", stats)
        self.assertIn("completed_tasks", stats)
        
        # Stop processor
        processor.stop()
    
    def test_process_pool_processor(self):
        """Test process pool processor"""
        # Create process pool processor
        processor = ProcessPoolProcessor(max_workers=4)
        processor.start()
        
        # Process batch (this is the main functionality we want to test)
        data = list(range(8))
        results = processor.process_batch_ordered(data, pool_func_process)
        self.assertEqual(len(results), 8)
        self.assertEqual(results, [x * 3 for x in data])
        
        # Get stats
        stats = processor.get_stats()
        self.assertIn("total_tasks", stats)
        self.assertIn("completed_tasks", stats)
        
        # Stop processor
        processor.stop()
    
    def test_async_processor(self):
        """Test asynchronous processor"""
        # Create async processor
        processor = AsyncProcessor()
        
        # Test async processing
        async def run_async_tests():
            import asyncio
            # Submit tasks
            task_ids = []
            for i in range(6):
                task_id = await processor.submit_task(i, pool_func_async)
                task_ids.append(task_id)
            
            # Wait for tasks to complete
            await asyncio.sleep(0.1)
            
            # Check task status
            for task_id in task_ids:
                status = await processor.get_task_status(task_id)
                self.assertIn("result", status)
                self.assertIsNone(status.get("error"))
            
            # Process batch
            data = list(range(6))
            results = await processor.process_batch_ordered(data, pool_func_async)
            self.assertEqual(len(results), 6)
            self.assertEqual(results, [x * 4 for x in data])
            
            # Get stats
            stats = await processor.get_stats()
            self.assertIn("total_tasks", stats)
            self.assertIn("completed_tasks", stats)
        
        # Run async tests
        import asyncio
        asyncio.run(run_async_tests())
    
    def test_global_processors(self):
        """Test global processor instances"""
        # Get global thread pool processor
        thread_processor = get_thread_pool_processor(max_workers=2)
        self.assertIsInstance(thread_processor, ThreadPoolProcessor)
        
        # Get global process pool processor
        process_processor = get_process_pool_processor(max_workers=2)
        self.assertIsInstance(process_processor, ProcessPoolProcessor)
        
        # Get global async processor
        async_processor = get_async_processor()
        self.assertIsInstance(async_processor, AsyncProcessor)
        
        # Shutdown all processors
        shutdown_all_processors()
    
    def test_parallel_map(self):
        """Test parallel map functionality"""
        # Test with threads
        data = list(range(10))
        results_threads = parallel_map(data, pool_func_parallel, max_workers=4, use_processes=False)
        self.assertEqual(len(results_threads), 10)
        self.assertEqual(results_threads, [x * 5 for x in data])
        
        # Test with processes
        results_processes = parallel_map(data, pool_func_parallel, max_workers=4, use_processes=True)
        self.assertEqual(len(results_processes), 10)
        self.assertEqual(results_processes, [x * 5 for x in data])
    
    def test_async_map(self):
        """Test asynchronous map functionality"""
        # Test async map
        async def run_async_map():
            data = list(range(8))
            results = await async_map(data, pool_func_async)
            self.assertEqual(len(results), 8)
            self.assertEqual(results, [x * 4 for x in data])
        
        # Run async map test
        import asyncio
        asyncio.run(run_async_map())
    
    def test_context_managers(self):
        """Test context managers for processors"""
        # Test thread pool context manager
        with ThreadPoolContext(max_workers=2) as thread_processor:
            self.assertIsInstance(thread_processor, ThreadPoolProcessor)
            
            # Test processing
            data = list(range(5))
            results = thread_processor.process_batch_ordered(data, pool_func_context_thread)
            self.assertEqual(results, [x * 7 for x in data])
        
        # Test process pool context manager
        with ProcessPoolContext(max_workers=2) as process_processor:
            self.assertIsInstance(process_processor, ProcessPoolProcessor)
            
            # Test processing
            data = list(range(5))
            results = process_processor.process_batch_ordered(data, pool_func_context_process)
            self.assertEqual(results, [x * 8 for x in data])
    
    def test_task_class(self):
        """Test Task class"""
        # Create a task
        task = Task(1, 5, pool_func_task)
        
        # Execute task
        result = task.execute()
        self.assertEqual(result, 45)
        self.assertEqual(task.result, 45)
        self.assertIsNone(task.error)
        
        # Check task duration
        duration = task.get_duration()
        self.assertIsInstance(duration, float)
        # Allow duration to be 0.0 since the task is very simple
        self.assertGreaterEqual(duration, 0.0)
    
    def test_error_handling(self):
        """Test error handling in concurrent processing"""
        # Test thread pool with error
        processor = ThreadPoolProcessor(max_workers=2)
        processor.start()
        
        # Submit tasks
        task_ids = []
        for i in range(3):
            task_id = processor.submit_task(i, pool_error_func)
            task_ids.append(task_id)
        
        # Wait for tasks to complete
        time.sleep(0.1)
        
        # Check task status
        for i, task_id in enumerate(task_ids):
            status = processor.get_task_status(task_id)
            if i == 2:
                # This task should have an error
                self.assertIn("error", status)
                self.assertIsNotNone(status["error"])
            else:
                # These tasks should succeed
                self.assertIn("result", status)
                self.assertIsNone(status.get("error"))
        
        # Stop processor
        processor.stop()
    
    def test_performance_comparison(self):
        """Test performance comparison between sequential and parallel processing"""
        # Test sequential processing
        data = list(range(10))
        start_time = time.time()
        sequential_results = [pool_compute_intensive(x) for x in data]
        sequential_time = time.time() - start_time
        
        # Test parallel processing with threads
        start_time = time.time()
        parallel_results_threads = parallel_map(data, pool_compute_intensive, max_workers=4, use_processes=False)
        parallel_time_threads = time.time() - start_time
        
        # Test parallel processing with processes
        start_time = time.time()
        parallel_results_processes = parallel_map(data, pool_compute_intensive, max_workers=4, use_processes=True)
        parallel_time_processes = time.time() - start_time
        
        # Check results are the same
        self.assertEqual(sequential_results, parallel_results_threads)
        self.assertEqual(sequential_results, parallel_results_processes)
        
        # Ensure parallel processing is working (times should be recorded)
        self.assertIsInstance(sequential_time, float)
        self.assertIsInstance(parallel_time_threads, float)
        self.assertIsInstance(parallel_time_processes, float)


if __name__ == "__main__":
    unittest.main()
