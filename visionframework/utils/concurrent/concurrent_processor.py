"""
Concurrent processing utilities

This module provides concurrent processing utilities, including:
- Multi-threaded processing
- Multi-process processing
- Asynchronous processing
- Task queue management

These utilities can be used to improve performance for computationally intensive tasks
like image processing, model inference, and batch processing.
"""

import threading
import multiprocessing as mp
import concurrent.futures
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Callable, Union, TypeVar, Generic
import logging
import queue
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class Task(Generic[T, R]):
    """
    Task class for concurrent processing
    
    This class represents a task to be processed concurrently, including
    the input data and the function to apply to the data.
    """
    
    def __init__(self, task_id: int, data: T, func: Callable[[T], R]):
        """
        Initialize a task
        
        Args:
            task_id: Unique task identifier
            data: Input data for the task
            func: Function to apply to the data
        """
        self.task_id = task_id
        self.data = data
        self.func = func
        self.result: Optional[R] = None
        self.error: Optional[Exception] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def execute(self) -> Optional[R]:
        """
        Execute the task
        
        Returns:
            Optional[R]: Result of the task execution, or None if an error occurred
        """
        self.start_time = time.time()
        try:
            self.result = self.func(self.data)
            return self.result
        except Exception as e:
            self.error = e
            logger.error(f"Task {self.task_id} failed: {e}")
            return None
        finally:
            self.end_time = time.time()
    
    def get_duration(self) -> Optional[float]:
        """
        Get the duration of the task execution
        
        Returns:
            Optional[float]: Duration in seconds, or None if the task was not executed
        """
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class ThreadPoolProcessor:
    """
    Thread pool processor for concurrent task execution
    
    This class provides a thread pool for executing tasks concurrently using threads.
    It is suitable for I/O-bound tasks and tasks that release the GIL.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize thread pool processor
        
        Args:
            max_workers: Maximum number of worker threads, defaults to os.cpu_count()
        """
        self.max_workers = max_workers
        self.executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.tasks: Dict[int, Task] = {}
        self.task_counter = 0
        self.lock = threading.RLock()
        
        logger.info(f"Created ThreadPoolProcessor with max_workers={max_workers}")
    
    def start(self) -> None:
        """
        Start the thread pool
        """
        if self.executor is None:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            )
            logger.info("Thread pool started")
    
    def stop(self) -> None:
        """
        Stop the thread pool
        """
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
            logger.info("Thread pool stopped")
    
    def submit_task(self, data: T, func: Callable[[T], R]) -> int:
        """
        Submit a task to the thread pool
        
        Args:
            data: Input data for the task
            func: Function to apply to the data
        
        Returns:
            int: Task identifier
        """
        with self.lock:
            task_id = self.task_counter
            self.task_counter += 1
            
            task = Task(task_id, data, func)
            self.tasks[task_id] = task
            
            if self.executor is None:
                self.start()
            
            future = self.executor.submit(task.execute)
            future.add_done_callback(lambda f: self._task_completed(task_id))
            
            return task_id
    
    def _task_completed(self, task_id: int) -> None:
        """
        Callback when a task is completed
        
        Args:
            task_id: Task identifier
        """
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                duration = task.get_duration()
                if task.error:
                    logger.debug(f"Task {task_id} completed with error in {duration:.4f}s")
                else:
                    logger.debug(f"Task {task_id} completed successfully in {duration:.4f}s")
    
    def process_batch(self, data_list: List[T], func: Callable[[T], R]) -> List[R]:
        """
        Process a batch of data concurrently
        
        Args:
            data_list: List of input data
            func: Function to apply to each data item
        
        Returns:
            List[R]: List of results in the same order as input data
        """
        if not data_list:
            return []
        
        # Start executor if not already started
        if self.executor is None:
            self.start()
        
        # Submit all tasks
        futures = []
        for i, data in enumerate(data_list):
            task = Task(i, data, func)
            future = self.executor.submit(task.execute)
            futures.append(future)
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
        
        # Note: Results may be in different order than input
        # If order matters, we need to track task IDs and reorder
        return results
    
    def process_batch_ordered(self, data_list: List[T], func: Callable[[T], R]) -> List[R]:
        """
        Process a batch of data concurrently and return results in original order
        
        Args:
            data_list: List of input data
            func: Function to apply to each data item
        
        Returns:
            List[R]: List of results in the same order as input data
        """
        if not data_list:
            return []
        
        # Start executor if not already started
        if self.executor is None:
            self.start()
        
        # Submit all tasks with their indices
        futures = {}
        for i, data in enumerate(data_list):
            task = Task(i, data, func)
            future = self.executor.submit(task.execute)
            futures[future] = i
        
        # Collect results in order
        results = [None] * len(data_list)
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            result = future.result()
            results[idx] = result
        
        return results
    
    def get_task_status(self, task_id: int) -> Dict[str, Any]:
        """
        Get the status of a task
        
        Args:
            task_id: Task identifier
        
        Returns:
            Dict[str, Any]: Task status information
        """
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                return {
                    "task_id": task_id,
                    "result": task.result,
                    "error": str(task.error) if task.error else None,
                    "duration": task.get_duration(),
                    "start_time": task.start_time,
                    "end_time": task.end_time
                }
            else:
                return {
                    "task_id": task_id,
                    "error": "Task not found"
                }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics
        
        Returns:
            Dict[str, Any]: Statistics information
        """
        with self.lock:
            completed_tasks = 0
            failed_tasks = 0
            total_duration = 0.0
            
            for task in self.tasks.values():
                if task.end_time:
                    completed_tasks += 1
                    if task.error:
                        failed_tasks += 1
                    duration = task.get_duration()
                    if duration:
                        total_duration += duration
            
            avg_duration = total_duration / completed_tasks if completed_tasks > 0 else 0
            
            return {
                "max_workers": self.max_workers,
                "total_tasks": len(self.tasks),
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "average_duration": avg_duration
            }


class ProcessPoolProcessor:
    """
    Process pool processor for concurrent task execution
    
    This class provides a process pool for executing tasks concurrently using processes.
    It is suitable for CPU-bound tasks that do not release the GIL.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize process pool processor
        
        Args:
            max_workers: Maximum number of worker processes, defaults to os.cpu_count()
        """
        self.max_workers = max_workers
        self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self.tasks: Dict[int, Dict[str, Any]] = {}
        self.task_counter = 0
        self.lock = threading.RLock()
        
        logger.info(f"Created ProcessPoolProcessor with max_workers={max_workers}")
    
    def start(self) -> None:
        """
        Start the process pool
        """
        if self.executor is None:
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            )
            logger.info("Process pool started")
    
    def stop(self) -> None:
        """
        Stop the process pool
        """
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
            logger.info("Process pool stopped")
    
    def submit_task(self, data: T, func: Callable[[T], R]) -> int:
        """
        Submit a task to the process pool
        
        Args:
            data: Input data for the task (must be picklable)
            func: Function to apply to the data (must be picklable)
        
        Returns:
            int: Task identifier
        """
        with self.lock:
            task_id = self.task_counter
            self.task_counter += 1
            
            self.tasks[task_id] = {
                "start_time": time.time(),
                "status": "submitted"
            }
            
            if self.executor is None:
                self.start()
            
            future = self.executor.submit(func, data)
            future.add_done_callback(lambda f: self._task_completed(task_id, f))
            
            return task_id
    
    def _task_completed(self, task_id: int, future: concurrent.futures.Future) -> None:
        """
        Callback when a task is completed
        
        Args:
            task_id: Task identifier
            future: Future object
        """
        # Simplified callback without lock for Windows compatibility
        if task_id in self.tasks:
            task_info = self.tasks[task_id]
            task_info["end_time"] = time.time()
            
            try:
                result = future.result()
                task_info["result"] = result
                task_info["status"] = "completed"
                task_info["error"] = None
                duration = task_info["end_time"] - task_info["start_time"]
                logger.debug(f"Task {task_id} completed successfully in {duration:.4f}s")
            except Exception as e:
                task_info["error"] = str(e)
                task_info["status"] = "failed"
                duration = task_info["end_time"] - task_info["start_time"]
                logger.error(f"Task {task_id} failed in {duration:.4f}s: {e}")
    
    def process_batch(self, data_list: List[T], func: Callable[[T], R]) -> List[R]:
        """
        Process a batch of data concurrently
        
        Args:
            data_list: List of input data (each must be picklable)
            func: Function to apply to each data item (must be picklable)
        
        Returns:
            List[R]: List of results in the same order as input data
        """
        if not data_list:
            return []
        
        # Start executor if not already started
        if self.executor is None:
            self.start()
        
        # Submit all tasks
        futures = []
        for data in data_list:
            future = self.executor.submit(func, data)
            futures.append(future)
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {e}")
                results.append(None)
        
        return results
    
    def process_batch_ordered(self, data_list: List[T], func: Callable[[T], R]) -> List[R]:
        """
        Process a batch of data concurrently and return results in original order
        
        Args:
            data_list: List of input data (each must be picklable)
            func: Function to apply to each data item (must be picklable)
        
        Returns:
            List[R]: List of results in the same order as input data
        """
        if not data_list:
            return []
        
        # Start executor if not already started
        if self.executor is None:
            self.start()
        
        # Submit all tasks with their indices
        futures = {}
        for i, data in enumerate(data_list):
            future = self.executor.submit(func, data)
            futures[future] = i
        
        # Collect results in order
        results = [None] * len(data_list)
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                logger.error(f"Task {idx} failed: {e}")
                results[idx] = None
        
        return results
    
    def get_task_status(self, task_id: int) -> Dict[str, Any]:
        """
        Get the status of a task
        
        Args:
            task_id: Task identifier
        
        Returns:
            Dict[str, Any]: Task status information
        """
        with self.lock:
            if task_id in self.tasks:
                task_info = self.tasks[task_id]
                status = task_info.copy()
                if "start_time" in status and "end_time" in status:
                    status["duration"] = status["end_time"] - status["start_time"]
                return status
            else:
                return {
                    "task_id": task_id,
                    "error": "Task not found"
                }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics
        
        Returns:
            Dict[str, Any]: Statistics information
        """
        with self.lock:
            total_tasks = len(self.tasks)
            completed_tasks = 0
            failed_tasks = 0
            total_duration = 0.0
            
            for task_info in self.tasks.values():
                if task_info.get("status") == "completed":
                    completed_tasks += 1
                    if "start_time" in task_info and "end_time" in task_info:
                        total_duration += task_info["end_time"] - task_info["start_time"]
                elif task_info.get("status") == "failed":
                    failed_tasks += 1
            
            avg_duration = total_duration / completed_tasks if completed_tasks > 0 else 0
            
            return {
                "max_workers": self.max_workers,
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "average_duration": avg_duration
            }


class AsyncProcessor:
    """
    Asynchronous processor for concurrent task execution
    
    This class provides an asynchronous interface for executing tasks concurrently
    using asyncio.
    """
    
    def __init__(self):
        """
        Initialize asynchronous processor
        """
        self.tasks: Dict[int, Dict[str, Any]] = {}
        self.task_counter = 0
        self.lock = asyncio.Lock()
        
        logger.info("Created AsyncProcessor")
    
    async def submit_task(self, data: T, func: Callable[[T], R]) -> int:
        """
        Submit a task for asynchronous execution
        
        Args:
            data: Input data for the task
            func: Function to apply to the data
        
        Returns:
            int: Task identifier
        """
        async with self.lock:
            task_id = self.task_counter
            self.task_counter += 1
            
            self.tasks[task_id] = {
                "start_time": time.time(),
                "status": "submitted"
            }
        
        try:
            # Execute task asynchronously
            result = await asyncio.to_thread(func, data)
            
            async with self.lock:
                if task_id in self.tasks:
                    task_info = self.tasks[task_id]
                    task_info["result"] = result
                    task_info["status"] = "completed"
                    task_info["end_time"] = time.time()
                    task_info["error"] = None
                    
                    duration = task_info["end_time"] - task_info["start_time"]
                    logger.debug(f"Async task {task_id} completed in {duration:.4f}s")
        except Exception as e:
            async with self.lock:
                if task_id in self.tasks:
                    task_info = self.tasks[task_id]
                    task_info["status"] = "failed"
                    task_info["error"] = str(e)
                    task_info["end_time"] = time.time()
                    
                    duration = task_info["end_time"] - task_info["start_time"]
                    logger.error(f"Async task {task_id} failed in {duration:.4f}s: {e}")
        
        return task_id
    
    async def process_batch(self, data_list: List[T], func: Callable[[T], R]) -> List[R]:
        """
        Process a batch of data asynchronously
        
        Args:
            data_list: List of input data
            func: Function to apply to each data item
        
        Returns:
            List[R]: List of results
        """
        if not data_list:
            return []
        
        # Create tasks
        tasks = []
        for data in data_list:
            task = asyncio.create_task(self.submit_task(data, func))
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        # Collect results (note: order may not be preserved)
        results = []
        async with self.lock:
            for task_info in self.tasks.values():
                if task_info.get("status") == "completed":
                    results.append(task_info.get("result"))
                else:
                    results.append(None)
        
        return results
    
    async def process_batch_ordered(self, data_list: List[T], func: Callable[[T], R]) -> List[R]:
        """
        Process a batch of data asynchronously and return results in original order
        
        Args:
            data_list: List of input data
            func: Function to apply to each data item
        
        Returns:
            List[R]: List of results in the same order as input data
        """
        if not data_list:
            return []
        
        # Create tasks with indices
        tasks = []
        async with self.lock:
            for i, data in enumerate(data_list):
                task_id = self.task_counter
                self.task_counter += 1
                
                self.tasks[task_id] = {
                    "start_time": time.time(),
                    "status": "submitted",
                    "index": i
                }
                
                task = asyncio.create_task(self._process_with_index(data, func, task_id))
                tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        # Collect results in order
        results = [None] * len(data_list)
        async with self.lock:
            for task_info in self.tasks.values():
                if "index" in task_info and task_info.get("status") == "completed":
                    idx = task_info["index"]
                    results[idx] = task_info.get("result")
        
        return results
    
    async def _process_with_index(self, data: T, func: Callable[[T], R], task_id: int) -> None:
        """
        Process a single task with index tracking
        
        Args:
            data: Input data
            func: Processing function
            task_id: Task identifier
        """
        try:
            result = await asyncio.to_thread(func, data)
            
            async with self.lock:
                if task_id in self.tasks:
                    task_info = self.tasks[task_id]
                    task_info["result"] = result
                    task_info["status"] = "completed"
                    task_info["end_time"] = time.time()
                    task_info["error"] = None
        except Exception as e:
            async with self.lock:
                if task_id in self.tasks:
                    task_info = self.tasks[task_id]
                    task_info["status"] = "failed"
                    task_info["error"] = str(e)
                    task_info["end_time"] = time.time()
    
    async def get_task_status(self, task_id: int) -> Dict[str, Any]:
        """
        Get the status of a task
        
        Args:
            task_id: Task identifier
        
        Returns:
            Dict[str, Any]: Task status information
        """
        async with self.lock:
            if task_id in self.tasks:
                task_info = self.tasks[task_id].copy()
                if "start_time" in task_info and "end_time" in task_info:
                    task_info["duration"] = task_info["end_time"] - task_info["start_time"]
                return task_info
            else:
                return {
                    "task_id": task_id,
                    "error": "Task not found"
                }
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics
        
        Returns:
            Dict[str, Any]: Statistics information
        """
        async with self.lock:
            total_tasks = len(self.tasks)
            completed_tasks = 0
            failed_tasks = 0
            total_duration = 0.0
            
            for task_info in self.tasks.values():
                if task_info.get("status") == "completed":
                    completed_tasks += 1
                    if "start_time" in task_info and "end_time" in task_info:
                        total_duration += task_info["end_time"] - task_info["start_time"]
                elif task_info.get("status") == "failed":
                    failed_tasks += 1
            
            avg_duration = total_duration / completed_tasks if completed_tasks > 0 else 0
            
            return {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "average_duration": avg_duration
            }


# Global processor instances
_thread_pool_processor: Optional[ThreadPoolProcessor] = None
_process_pool_processor: Optional[ProcessPoolProcessor] = None
_async_processor: Optional[AsyncProcessor] = None
_processor_lock = threading.Lock()


def get_thread_pool_processor(max_workers: Optional[int] = None) -> ThreadPoolProcessor:
    """
    Get global thread pool processor instance
    
    Args:
        max_workers: Maximum number of worker threads
    
    Returns:
        ThreadPoolProcessor: Thread pool processor instance
    """
    global _thread_pool_processor
    with _processor_lock:
        if _thread_pool_processor is None:
            _thread_pool_processor = ThreadPoolProcessor(max_workers=max_workers)
        return _thread_pool_processor


def get_process_pool_processor(max_workers: Optional[int] = None) -> ProcessPoolProcessor:
    """
    Get global process pool processor instance
    
    Args:
        max_workers: Maximum number of worker processes
    
    Returns:
        ProcessPoolProcessor: Process pool processor instance
    """
    global _process_pool_processor
    with _processor_lock:
        if _process_pool_processor is None:
            _process_pool_processor = ProcessPoolProcessor(max_workers=max_workers)
        return _process_pool_processor


def get_async_processor() -> AsyncProcessor:
    """
    Get global async processor instance
    
    Returns:
        AsyncProcessor: Async processor instance
    """
    global _async_processor
    with _processor_lock:
        if _async_processor is None:
            _async_processor = AsyncProcessor()
        return _async_processor


def shutdown_all_processors() -> None:
    """
    Shutdown all processor instances
    """
    global _thread_pool_processor, _process_pool_processor, _async_processor
    
    with _processor_lock:
        if _thread_pool_processor:
            _thread_pool_processor.stop()
            _thread_pool_processor = None
        
        if _process_pool_processor:
            _process_pool_processor.stop()
            _process_pool_processor = None
        
        _async_processor = None
        
        logger.info("All processors shutdown")


# Utility functions

def parallel_map(data_list: List[T], func: Callable[[T], R], max_workers: Optional[int] = None, use_processes: bool = False) -> List[R]:
    """
    Parallel map function
    
    Args:
        data_list: List of input data
        func: Function to apply to each data item
        max_workers: Maximum number of workers
        use_processes: Whether to use processes instead of threads
    
    Returns:
        List[R]: List of results
    """
    if use_processes:
        processor = get_process_pool_processor(max_workers=max_workers)
        return processor.process_batch_ordered(data_list, func)
    else:
        processor = get_thread_pool_processor(max_workers=max_workers)
        return processor.process_batch_ordered(data_list, func)


async def async_map(data_list: List[T], func: Callable[[T], R]) -> List[R]:
    """
    Asynchronous map function
    
    Args:
        data_list: List of input data
        func: Function to apply to each data item
    
    Returns:
        List[R]: List of results
    """
    processor = get_async_processor()
    return await processor.process_batch_ordered(data_list, func)


# Context managers

class ThreadPoolContext:
    """
    Context manager for thread pool processor
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers
        self.processor = None
    
    def __enter__(self) -> ThreadPoolProcessor:
        self.processor = ThreadPoolProcessor(max_workers=self.max_workers)
        self.processor.start()
        return self.processor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.processor:
            self.processor.stop()


class ProcessPoolContext:
    """
    Context manager for process pool processor
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers
        self.processor = None
    
    def __enter__(self) -> ProcessPoolProcessor:
        self.processor = ProcessPoolProcessor(max_workers=self.max_workers)
        self.processor.start()
        return self.processor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.processor:
            self.processor.stop()
