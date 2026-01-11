"""
RF-DETR 检测器使用示例

RF-DETR 是 Roboflow 开发的高性能实时目标检测模型。
本示例展示如何使用 RF-DETR 检测器进行目标检测。

RF-DETR 特点：
- 高性能实时检测
- 支持多种预训练模型
- 易于集成到现有流程中
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from visionframework import Detector, Visualizer


def example_basic_detection():
    """
    示例 1: RF-DETR 基本检测
    
    本示例展示如何使用 RF-DETR 检测器进行基本的目标检测。
    适用于需要快速、准确检测的场景。
    """
    print("=" * 70)
    print("示例 1: RF-DETR 基本检测")
    print("=" * 70)
    
    # ========== 步骤 1: 初始化 RF-DETR 检测器 ==========
    print("\n1. 初始化 RF-DETR 检测器...")
    
    # RF-DETR 检测器配置
    detector = Detector({
        "model_type": "rfdetr",        # 指定使用 RF-DETR 模型
        "conf_threshold": 0.5,         # 置信度阈值（50%）
        "device": "cpu"                # 设备类型：可选 "cpu", "cuda", "mps"
    })
    
    # 初始化检测器
    if not detector.initialize():
        print("✗ 检测器初始化失败！")
        print("  提示: 请确保已安装 rfdetr: pip install rfdetr supervision")
        return
    
    print("  ✓ RF-DETR 检测器初始化成功")
    
    # ========== 步骤 2: 准备测试图像 ==========
    print("\n2. 准备测试图像...")
    
    # 创建一个简单的测试图像（实际使用时可以加载真实图像）
    test_image = np.zeros((640, 480, 3), dtype=np.uint8)
    test_image[:] = (128, 128, 128)  # 灰色背景
    
    # 添加一些彩色矩形作为测试对象
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), -1)
    cv2.rectangle(test_image, (300, 150), (400, 250), (255, 0, 0), -1)
    
    print("  ✓ 测试图像已创建")
    
    # ========== 步骤 3: 运行检测 ==========
    print("\n3. 运行检测...")
    
    # detect() 方法返回 Detection 对象列表
    detections = detector.detect(test_image)
    
    print(f"  ✓ 检测完成，发现 {len(detections)} 个对象")
    
    # ========== 步骤 4: 显示检测结果详情 ==========
    if detections:
        print("\n  检测结果详情:")
        for i, det in enumerate(detections):
            print(f"    对象 {i+1}:")
            print(f"      类别: {det.class_name}")
            print(f"      置信度: {det.confidence:.2f}")
            print(f"      边界框: {det.bbox}")
    
    # ========== 步骤 5: 可视化结果 ==========
    print("\n4. 可视化结果...")
    
    # 创建可视化器
    visualizer = Visualizer({
        "show_labels": True,
        "show_confidences": True
    })
    
    # 绘制检测结果
    result_image = visualizer.draw_detections(test_image, detections)
    
    # ========== 步骤 6: 保存结果 ==========
    output_path = "rfdetr_output.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"  ✓ 结果已保存到: {output_path}")
    
    # 尝试显示结果（如果环境支持 GUI）
    try:
        cv2.imshow("RF-DETR Detection", result_image)
        print("\n  提示: 按任意键关闭窗口")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("  提示: 无法显示图像窗口（可能在没有 GUI 的环境中运行）")


def example_with_real_image(image_path):
    """
    示例 2: 使用真实图像进行检测
    
    本示例展示如何使用 RF-DETR 检测器处理真实图像。
    
    Args:
        image_path: 图像文件路径
    """
    print("=" * 70)
    print("示例 2: RF-DETR 真实图像检测")
    print("=" * 70)
    
    # ========== 步骤 1: 加载图像 ==========
    print(f"\n1. 加载图像: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"✗ 无法加载图像: {image_path}")
        print("  提示: 请提供有效的图像路径")
        return
    
    print(f"  ✓ 图像加载成功")
    print(f"    图像尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # ========== 步骤 2: 初始化检测器 ==========
    print("\n2. 初始化 RF-DETR 检测器...")
    
    detector = Detector({
        "model_type": "rfdetr",
        "conf_threshold": 0.5,  # 可以根据需要调整阈值
        "device": "cpu"
    })
    
    if not detector.initialize():
        print("✗ 检测器初始化失败！")
        return
    
    print("  ✓ 检测器初始化成功")
    
    # ========== 步骤 3: 运行检测 ==========
    print("\n3. 运行检测...")
    
    detections = detector.detect(image)
    print(f"  ✓ 检测完成，发现 {len(detections)} 个对象")
    
    # ========== 步骤 4: 按置信度排序并显示前几个结果 ==========
    if detections:
        # 按置信度降序排序
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        print("\n  前5个检测结果（按置信度排序）:")
        for i, det in enumerate(detections[:5]):
            print(f"    {i+1}. {det.class_name}: {det.confidence:.2f}")
    
    # ========== 步骤 5: 可视化结果 ==========
    print("\n4. 可视化结果...")
    
    visualizer = Visualizer({
        "show_labels": True,
        "show_confidences": True,
        "line_thickness": 2
    })
    
    result_image = visualizer.draw_detections(image, detections)
    
    # ========== 步骤 6: 保存结果 ==========
    output_path = "rfdetr_real_output.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"  ✓ 结果已保存到: {output_path}")


def example_pipeline_integration():
    """
    示例 3: 与 VisionPipeline 集成
    
    本示例展示如何将 RF-DETR 检测器集成到 VisionPipeline 中，
    实现检测和跟踪的完整流程。
    """
    print("=" * 70)
    print("示例 3: RF-DETR 管道集成")
    print("=" * 70)
    
    from visionframework import VisionPipeline
    
    # ========== 步骤 1: 创建使用 RF-DETR 的视觉管道 ==========
    print("\n1. 创建视觉管道...")
    
    # VisionPipeline 可以统一管理检测器和跟踪器
    pipeline = VisionPipeline({
        "detector_config": {
            "model_type": "rfdetr",      # 使用 RF-DETR 检测器
            "conf_threshold": 0.5,
            "device": "cpu"
        },
        "tracker_config": {
            "max_age": 30,              # 目标丢失后保留的最大帧数
            "min_hits": 3              # 确认跟踪所需的最小命中次数
        },
        "enable_tracking": True         # 启用跟踪功能
    })
    
    if not pipeline.initialize():
        print("✗ 管道初始化失败！")
        print("  提示: 请确保已安装 rfdetr: pip install rfdetr supervision")
        return
    
    print("  ✓ 视觉管道初始化成功")
    
    # ========== 步骤 2: 准备测试图像 ==========
    print("\n2. 准备测试图像...")
    
    test_image = np.zeros((640, 480, 3), dtype=np.uint8)
    test_image[:] = (128, 128, 128)
    
    # 添加测试对象
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), -1)
    cv2.rectangle(test_image, (300, 150), (400, 250), (255, 0, 0), -1)
    
    print("  ✓ 测试图像已创建")
    
    # ========== 步骤 3: 处理图像 ==========
    print("\n3. 处理图像...")
    
    # process() 方法会同时进行检测和跟踪（如果启用）
    results = pipeline.process(test_image)
    
    detections = results["detections"]
    tracks = results["tracks"]
    
    print(f"  ✓ 处理完成")
    print(f"    检测结果: {len(detections)} 个对象")
    print(f"    跟踪结果: {len(tracks)} 个轨迹")
    
    # ========== 步骤 4: 可视化结果 ==========
    print("\n4. 可视化结果...")
    
    visualizer = Visualizer({
        "show_labels": True,
        "show_confidences": True,
        "show_track_ids": True
    })
    
    # 可以同时绘制检测和跟踪结果
    result_image = visualizer.draw_results(
        test_image,
        detections=detections,
        tracks=tracks
    )
    
    # ========== 步骤 5: 保存结果 ==========
    output_path = "rfdetr_pipeline_output.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"  ✓ 结果已保存到: {output_path}")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("RF-DETR 检测器使用示例")
    print("=" * 70)
    print("\n本示例展示如何使用 RF-DETR 检测器进行目标检测")
    print("=" * 70)
    
    import sys
    
    try:
        # 运行基本检测示例
        example_basic_detection()
        
        print("\n" + "-" * 70)
        
        # 如果提供了图像路径，运行真实图像检测示例
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            example_with_real_image(image_path)
        else:
            print("\n提示: 可以传入图像路径作为参数来使用真实图像检测")
            print("例如: python rfdetr_example.py your_image.jpg")
        
        print("\n" + "-" * 70)
        
        # 运行管道集成示例
        example_pipeline_integration()
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

