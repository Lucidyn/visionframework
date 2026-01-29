#!/usr/bin/env python3
"""
内存池管理使用示例

本示例演示如何使用 Vision Framework 的内存池管理功能。
包括：
1. 创建和初始化内存池
2. 分配和释放内存块
3. 使用内存池进行批处理
4. 优化内存使用
5. 监控内存池状态
"""

import os
import sys
import cv2
import numpy as np
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionframework.utils.memory.memory_manager import MemoryManager, MemoryPool
from visionframework.core.pipeline import VisionPipeline


def test_basic_memory_pool():
    """
    测试基本的内存池功能
    """
    print("=== 测试基本内存池功能 ===")
    
    # 1. 创建内存池
    print("\n1. 创建内存池:")
    memory_pool = MemoryPool(
        block_shape=(480, 640, 3),  # 内存块形状
        dtype=np.uint8,             # 数据类型
        max_blocks=4,               # 最大内存块数量
        min_blocks=2,               # 最小内存块数量
        enable_dynamic_resizing=True,  # 启用动态调整大小
        resize_factor=1.5           # 调整大小因子
    )
    
    print("内存池创建成功")
    print(f"内存池状态: {memory_pool.get_status()}")
    
    # 2. 分配内存
    print("\n2. 分配内存:")
    memory1 = memory_pool.acquire()
    print(f"分配的内存1形状: {memory1.shape}")
    print(f"内存池状态: {memory_pool.get_status()}")
    
    memory2 = memory_pool.acquire()
    print(f"分配的内存2形状: {memory2.shape}")
    print(f"内存池状态: {memory_pool.get_status()}")
    
    # 3. 使用内存
    print("\n3. 使用内存:")
    # 填充内存
    memory1[:] = 255  # 填充白色
    memory2[:] = 0    # 填充黑色
    
    print("内存使用成功")
    print(f"内存1平均值: {np.mean(memory1):.2f}")
    print(f"内存2平均值: {np.mean(memory2):.2f}")
    
    # 4. 释放内存
    print("\n4. 释放内存:")
    memory_pool.release(memory1)
    print(f"释放内存1后状态: {memory_pool.get_status()}")
    
    memory_pool.release(memory2)
    print(f"释放内存2后状态: {memory_pool.get_status()}")
    
    # 5. 测试内存池优化
    print("\n5. 测试内存池优化:")
    memory_pool.optimize()
    print(f"优化后内存池状态: {memory_pool.get_status()}")
    
    return memory_pool


def test_global_memory_pool():
    """
    测试全局内存池功能
    """
    print("\n=== 测试全局内存池功能 ===")
    
    # 1. 获取全局内存池
    print("\n1. 获取全局内存池:")
    global_pool = MemoryManager.get_global_memory_pool()
    print(f"全局内存池状态: {global_pool.get_status()}")
    
    # 2. 初始化全局内存池
    print("\n2. 初始化全局内存池:")
    global_pool.initialize(
        min_blocks=4,
        block_size=(480, 640, 3),
        max_blocks=8
    )
    print(f"初始化后全局内存池状态: {global_pool.get_status()}")
    
    # 3. 分配和释放内存
    print("\n3. 分配和释放内存:")
    memory = global_pool.acquire()
    print(f"分配的内存形状: {memory.shape}")
    print(f"分配后全局内存池状态: {global_pool.get_status()}")
    
    global_pool.release(memory)
    print(f"释放后全局内存池状态: {global_pool.get_status()}")
    
    return global_pool


def test_memory_pool_for_batch_processing():
    """
    测试内存池在批处理中的使用
    """
    print("\n=== 测试内存池在批处理中的使用 ===")
    
    # 1. 创建内存池
    print("\n1. 创建批处理内存池:")
    batch_pool = MemoryPool(
        block_shape=(480, 640, 3),
        dtype=np.uint8,
        max_blocks=8,
        min_blocks=4
    )
    print(f"批处理内存池状态: {batch_pool.get_status()}")
    
    # 2. 创建测试图像
    print("\n2. 创建测试图像:")
    num_images = 6
    test_images = []
    for i in range(num_images):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (100 + i*50, 100), (300 + i*20, 300), (255, 0, 0), -1)
        test_images.append(image)
    print(f"创建了 {num_images} 张测试图像")
    
    # 3. 使用内存池进行批处理
    print("\n3. 使用内存池进行批处理:")
    start_time = time.time()
    
    processed_images = []
    memory_blocks = []
    
    # 分配内存
    for i in range(num_images):
        memory = batch_pool.acquire()
        memory_blocks.append(memory)
    
    print(f"分配了 {len(memory_blocks)} 个内存块")
    print(f"内存池状态: {batch_pool.get_status()}")
    
    # 处理图像
    for i, (image, memory) in enumerate(zip(test_images, memory_blocks)):
        # 将图像复制到内存块
        memory[:] = image
        
        # 处理图像（模拟）
        processed = memory.copy()
        cv2.putText(processed, f"Processed {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        processed_images.append(processed)
    
    # 释放内存
    for memory in memory_blocks:
        batch_pool.release(memory)
    
    end_time = time.time()
    print(f"批处理完成，耗时: {end_time - start_time:.4f}秒")
    print(f"处理后内存池状态: {batch_pool.get_status()}")
    print(f"处理了 {len(processed_images)} 张图像")
    
    return batch_pool


def test_memory_pool_with_pipeline():
    """
    测试内存池在 VisionPipeline 中的使用
    """
    print("\n=== 测试内存池在 VisionPipeline 中的使用 ===")
    
    # 1. 初始化全局内存池
    print("\n1. 初始化全局内存池:")
    global_pool = MemoryManager.get_global_memory_pool()
    global_pool.initialize(
        min_blocks=4,
        block_size=(480, 640, 3),
        max_blocks=8
    )
    print(f"全局内存池状态: {global_pool.get_status()}")
    
    # 2. 创建管道
    print("\n2. 创建 VisionPipeline:")
    config = {
        "detector_config": {
            "model_path": "yolov8n.pt",
            "device": "cpu"
        }
    }
    
    pipeline = VisionPipeline(config)
    pipeline.initialize()
    print("管道初始化成功")
    
    # 3. 创建测试图像
    print("\n3. 创建测试图像:")
    num_images = 4
    test_images = []
    for i in range(num_images):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (100 + i*100, 100), (200 + i*100, 200), (255, 255, 255), -1)
        test_images.append(image)
    print(f"创建了 {num_images} 张测试图像")
    
    # 4. 使用管道进行批处理
    print("\n4. 使用管道进行批处理:")
    start_time = time.time()
    
    results = pipeline.process_batch(test_images)
    
    end_time = time.time()
    print(f"批处理完成，耗时: {end_time - start_time:.4f}秒")
    print(f"处理了 {len(results)} 张图像")
    
    # 5. 打印结果
    for i, result in enumerate(results):
        print(f"  图像 {i+1} 检测到 {len(result['detections'])} 个目标")
    
    # 6. 检查内存池状态
    print("\n6. 检查内存池状态:")
    print(f"全局内存池状态: {global_pool.get_status()}")
    
    return pipeline


def test_memory_pool_optimization():
    """
    测试内存池优化功能
    """
    print("\n=== 测试内存池优化功能 ===")
    
    # 1. 创建内存池
    print("\n1. 创建内存池:")
    memory_pool = MemoryPool(
        block_shape=(480, 640, 3),
        dtype=np.uint8,
        max_blocks=10,
        min_blocks=2,
        enable_dynamic_resizing=True
    )
    print(f"初始内存池状态: {memory_pool.get_status()}")
    
    # 2. 分配大量内存
    print("\n2. 分配大量内存:")
    memory_blocks = []
    for i in range(8):
        memory = memory_pool.acquire()
        memory_blocks.append(memory)
        print(f"分配内存 {i+1} 后状态: {memory_pool.get_status()}")
    
    # 3. 释放部分内存
    print("\n3. 释放部分内存:")
    for i in range(5):
        memory_pool.release(memory_blocks[i])
        print(f"释放内存 {i+1} 后状态: {memory_pool.get_status()}")
    
    # 4. 优化内存池
    print("\n4. 优化内存池:")
    memory_pool.optimize()
    print(f"优化后内存池状态: {memory_pool.get_status()}")
    
    # 5. 释放剩余内存
    print("\n5. 释放剩余内存:")
    for i in range(5, 8):
        memory_pool.release(memory_blocks[i])
    print(f"释放所有内存后状态: {memory_pool.get_status()}")
    
    # 6. 再次优化
    print("\n6. 再次优化:")
    memory_pool.optimize()
    print(f"再次优化后内存池状态: {memory_pool.get_status()}")
    
    return memory_pool


def main():
    """
    主函数
    """
    print("=== 内存池管理示例 ===")
    
    try:
        # 测试基本内存池功能
        basic_pool = test_basic_memory_pool()
        
        # 测试全局内存池功能
        global_pool = test_global_memory_pool()
        
        # 测试内存池在批处理中的使用
        batch_pool = test_memory_pool_for_batch_processing()
        
        # 测试内存池在 VisionPipeline 中的使用
        pipeline = test_memory_pool_with_pipeline()
        
        # 测试内存池优化功能
        optimized_pool = test_memory_pool_optimization()
        
        print("\n=== 内存池管理示例完成 ===")
        print("所有测试通过！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
