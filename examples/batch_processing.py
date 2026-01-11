"""
批量图像处理示例

本示例展示如何批量处理多张图像，包括：
- 批量检测
- 结果汇总和统计
- 批量保存结果
- 进度显示
"""

import cv2
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from visionframework import Detector, Visualizer, ResultExporter


def batch_process_images(
    input_dir: str,
    output_dir: str = "batch_output",
    model_type: str = "yolo",
    conf_threshold: float = 0.25
):
    """
    批量处理图像
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出目录
        model_type: 模型类型 ("yolo", "detr", "rfdetr")
        conf_threshold: 置信度阈值
    """
    print("=" * 70)
    print("批量图像处理示例")
    print("=" * 70)
    
    # ========== 步骤 1: 准备输入和输出目录 ==========
    print(f"\n1. 准备输入和输出目录...")
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"✗ 输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    images_output = output_path / "images"  # 保存处理后的图像
    images_output.mkdir(exist_ok=True)
    
    print(f"  ✓ 输入目录: {input_dir}")
    print(f"  ✓ 输出目录: {output_dir}")
    
    # ========== 步骤 2: 查找所有图像文件 ==========
    print(f"\n2. 查找图像文件...")
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"✗ 在 {input_dir} 中未找到图像文件")
        return
    
    print(f"  ✓ 找到 {len(image_files)} 个图像文件")
    
    # ========== 步骤 3: 初始化检测器 ==========
    print(f"\n3. 初始化检测器 (模型类型: {model_type})...")
    
    detector_config = {
        "model_type": model_type,
        "conf_threshold": conf_threshold,
        "device": "cpu"
    }
    
    # 根据模型类型设置特定参数
    if model_type == "yolo":
        detector_config["model_path"] = "yolov8n.pt"
    elif model_type == "detr":
        detector_config["detr_model_name"] = "facebook/detr-resnet-50"
    elif model_type == "rfdetr":
        pass  # RF-DETR 使用默认模型
    
    detector = Detector(detector_config)
    
    if not detector.initialize():
        print(f"✗ 检测器初始化失败")
        return
    
    print("  ✓ 检测器初始化成功")
    
    # ========== 步骤 4: 初始化可视化器和导出器 ==========
    visualizer = Visualizer({
        "show_labels": True,
        "show_confidences": True
    })
    
    exporter = ResultExporter()
    
    # ========== 步骤 5: 批量处理图像 ==========
    print(f"\n4. 开始批量处理 {len(image_files)} 张图像...")
    print("-" * 70)
    
    all_results = []
    total_detections = 0
    
    for idx, image_file in enumerate(image_files, 1):
        # 显示进度
        progress = (idx / len(image_files)) * 100
        print(f"  处理 [{idx}/{len(image_files)}] ({progress:.1f}%): {image_file.name}")
        
        # 加载图像
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"    ✗ 无法加载图像: {image_file.name}")
            continue
        
        # 运行检测
        detections = detector.detect(image)
        total_detections += len(detections)
        
        # 可视化结果
        result_image = visualizer.draw_detections(image, detections)
        
        # 添加信息文本
        info_text = f"Detections: {len(detections)}"
        cv2.putText(
            result_image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # 保存处理后的图像
        output_image_path = images_output / f"processed_{image_file.name}"
        cv2.imwrite(str(output_image_path), result_image)
        
        # 保存结果信息
        result_info = {
            "image_file": image_file.name,
            "image_path": str(image_file),
            "image_size": {
                "width": image.shape[1],
                "height": image.shape[0]
            },
            "detection_count": len(detections),
            "detections": [det.to_dict() for det in detections],
            "output_image": str(output_image_path)
        }
        all_results.append(result_info)
        
        print(f"    ✓ 检测到 {len(detections)} 个对象")
    
    # ========== 步骤 6: 生成汇总报告 ==========
    print(f"\n5. 生成汇总报告...")
    
    # 统计信息
    total_images = len(all_results)
    avg_detections = total_detections / total_images if total_images > 0 else 0
    
    # 统计类别分布
    class_distribution = {}
    for result in all_results:
        for det in result["detections"]:
            class_name = det.get("class_name", "unknown")
            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
    
    # 创建汇总报告
    summary = {
        "total_images": total_images,
        "total_detections": total_detections,
        "average_detections_per_image": avg_detections,
        "class_distribution": class_distribution,
        "model_type": model_type,
        "conf_threshold": conf_threshold,
        "results": all_results
    }
    
    # 保存汇总报告
    summary_path = output_path / "batch_processing_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ 汇总报告已保存到: {summary_path}")
    
    # 导出检测结果为 CSV
    csv_path = output_path / "batch_detections.csv"
    
    # 准备 CSV 数据
    csv_data = []
    for result in all_results:
        for det in result["detections"]:
            csv_row = {
                "image_file": result["image_file"],
                "class_name": det.get("class_name", ""),
                "class_id": det.get("class_id", ""),
                "confidence": det.get("confidence", 0),
                "bbox_x1": det.get("bbox", [0, 0, 0, 0])[0],
                "bbox_y1": det.get("bbox", [0, 0, 0, 0])[1],
                "bbox_x2": det.get("bbox", [0, 0, 0, 0])[2],
                "bbox_y2": det.get("bbox", [0, 0, 0, 0])[3]
            }
            csv_data.append(csv_row)
    
    if csv_data:
        exporter.export_detections_to_csv(csv_data, str(csv_path))
        print(f"  ✓ CSV 结果已保存到: {csv_path}")
    
    # ========== 步骤 7: 打印统计信息 ==========
    print("\n" + "=" * 70)
    print("批量处理完成！")
    print("=" * 70)
    print(f"\n处理统计:")
    print(f"  总图像数: {total_images}")
    print(f"  总检测数: {total_detections}")
    print(f"  平均每张图像: {avg_detections:.2f} 个对象")
    print(f"\n类别分布:")
    for class_name, count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}")
    print(f"\n输出文件:")
    print(f"  处理后的图像: {images_output}")
    print(f"  汇总报告: {summary_path}")
    print(f"  CSV 结果: {csv_path}")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("批量图像处理示例")
    print("=" * 70)
    print("\n本示例展示如何批量处理多张图像")
    print("=" * 70)
    
    import sys
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print("  python batch_processing.py <input_dir> [output_dir] [model_type] [conf_threshold]")
        print("\n参数说明:")
        print("  input_dir: 输入图像目录（必需）")
        print("  output_dir: 输出目录（可选，默认: batch_output）")
        print("  model_type: 模型类型（可选，默认: yolo，可选: yolo/detr/rfdetr）")
        print("  conf_threshold: 置信度阈值（可选，默认: 0.25）")
        print("\n示例:")
        print("  python batch_processing.py ./images")
        print("  python batch_processing.py ./images ./output rfdetr 0.5")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "batch_output"
    model_type = sys.argv[3] if len(sys.argv) > 3 else "yolo"
    conf_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    
    try:
        batch_process_images(input_dir, output_dir, model_type, conf_threshold)
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

