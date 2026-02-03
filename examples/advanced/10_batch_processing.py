"""
10_batch_processing.py

批处理示例：
- 使用 BatchPipeline 进行批量图像处理
- 提高处理效率，特别适用于大规模图像处理
- 比较批处理和单张处理的性能差异

注意：
- 需要提前准备模型权重（如 yolov8n.pt），并放在当前工作目录或指定路径。
"""

import cv2
import numpy as np
import time

from visionframework import BatchPipeline, Visualizer


def _load_sample_images(count: int = 4) -> list:
    """
    加载示例图像。
    如果没有实际图像，生成一些随机图像。
    """
    images = []
    
    for i in range(count):
        # 尝试加载实际图像
        img_path = f"test_{i+1}.jpg"
        img = cv2.imread(img_path)
        
        # 如果没有实际图像，生成随机图像
        if img is None:
            # 生成 480x640 的随机图像
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            print(f"生成随机图像 {i+1}/{count}")
        else:
            print(f"加载图像 {i+1}/{count}: {img_path}")
        
        images.append(img)
    
    return images


def process_single_images(images: list, config: dict) -> tuple:
    """
    单张处理图像。
    """
    # 创建批处理管道（用于单张处理）
    pipeline = BatchPipeline(config)
    if not pipeline.initialize():
        print("BatchPipeline 初始化失败，请检查模型路径和依赖。")
        return [], 0
    
    results = []
    start_time = time.time()
    
    # 逐张处理
    for i, img in enumerate(images):
        result = pipeline.process([img])[0]  # 注意：BatchPipeline.process 总是返回列表
        results.append(result)
        print(f"单张处理进度: {i+1}/{len(images)}")
    
    elapsed_time = time.time() - start_time
    return results, elapsed_time


def process_batch_images(images: list, config: dict) -> tuple:
    """
    批量处理图像。
    """
    # 创建批处理管道
    batch_pipeline = BatchPipeline(config)
    if not batch_pipeline.initialize():
        print("BatchPipeline 初始化失败，请检查模型路径和依赖。")
        return [], 0
    
    start_time = time.time()
    
    # 批量处理
    results = batch_pipeline.process(images)
    
    elapsed_time = time.time() - start_time
    return results, elapsed_time


def visualize_results(images: list, results: list) -> None:
    """
    可视化处理结果。
    """
    visualizer = Visualizer()
    
    for i, (img, result) in enumerate(zip(images, results)):
        detections = result.get("detections", [])
        tracks = result.get("tracks", [])
        poses = result.get("poses", [])
        
        # 绘制结果
        vis_img = visualizer.draw_results(
            img.copy(),
            detections=detections,
            tracks=tracks,
            poses=poses
        )
        
        # 显示结果
        cv2.imshow(f"Result {i+1}", vis_img)
        print(f"显示结果 {i+1}: 检测到 {len(detections)} 个目标")
    
    print("按任意键关闭所有窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    # 1. 配置批处理管道
    config = {
        "detector_config": {
            "model_path": "yolov8n.pt",  # 可替换为你的模型路径
            "device": "auto",
            "conf_threshold": 0.25,
        },
        "enable_tracking": True,
        "enable_segmentation": False,
        "enable_pose_estimation": False,
        "batch_size": 4,  # 批处理大小，可根据 GPU 内存调整
    }

    # 2. 加载示例图像
    print("加载示例图像...")
    num_images = 8  # 处理 8 张图像
    images = _load_sample_images(num_images)
    
    if not images:
        print("没有图像可处理，请确保有测试图像或使用随机生成的图像。")
        return

    # 3. 单张处理
    print("\n开始单张处理...")
    single_results, single_time = process_single_images(images, config)
    print(f"单张处理完成，耗时: {single_time:.2f} 秒")
    print(f"平均每张图像耗时: {single_time / len(images):.2f} 秒")

    # 4. 批量处理
    print("\n开始批量处理...")
    batch_results, batch_time = process_batch_images(images, config)
    print(f"批量处理完成，耗时: {batch_time:.2f} 秒")
    print(f"平均每张图像耗时: {batch_time / len(images):.2f} 秒")

    # 5. 性能比较
    speedup = single_time / batch_time if batch_time > 0 else 0
    print(f"\n性能比较:")
    print(f"单张处理总耗时: {single_time:.2f} 秒")
    print(f"批量处理总耗时: {batch_time:.2f} 秒")
    print(f"速度提升: {speedup:.2f}x")

    # 6. 可视化批量处理结果
    print("\n可视化批量处理结果...")
    visualize_results(images, batch_results)


if __name__ == "__main__":
    main()
