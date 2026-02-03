"""
04_segmentation.py

基础分割示例：
- 使用 SAMSegmenter 对单张图片进行分割
- 支持点提示和边界框提示
- 可视化分割结果

注意：
- 需要提前准备 SAM 模型权重（如 sam_vit_b_01ec64.pth），并放在当前工作目录或指定路径。
"""

import cv2
import numpy as np

from visionframework import SAMSegmenter, Visualizer


def main() -> None:
    # 1. 创建分割器配置
    segmenter_config = {
        "model_path": "sam_vit_b_01ec64.pth",  # 可替换为你的模型路径
        "device": "auto",
        "conf_threshold": 0.5,
    }

    segmenter = SAMSegmenter(segmenter_config)

    # 2. 初始化模型（加载权重）
    if not segmenter.initialize():
        print("SAMSegmenter 初始化失败，请检查模型路径和依赖（segment-anything、torch 等）。")
        return

    # 3. 读取测试图片
    image_path = "test.jpg"  # 请替换为你自己的图片路径
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    # 4. 示例 1: 使用点提示进行分割
    print("示例 1: 使用点提示进行分割")
    
    # 定义点提示（可以根据实际图像内容调整）
    points = [(320, 240), (400, 300)]  # 图像中的点坐标
    point_labels = [1, 1]  # 1 表示前景点
    
    # 进行分割
    segments_with_points = segmenter.process(image, points=points, point_labels=point_labels)
    print(f"点提示分割结果数量: {len(segments_with_points)}")
    
    # 可视化结果
    visualizer = Visualizer()
    vis_image_points = visualizer.draw_segments(image.copy(), segments_with_points)
    
    # 绘制提示点
    for point, label in zip(points, point_labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(vis_image_points, point, 5, color, -1)

    # 5. 示例 2: 使用边界框提示进行分割
    print("示例 2: 使用边界框提示进行分割")
    
    # 定义边界框（可以根据实际图像内容调整）
    bbox = (200, 150, 450, 350)  # (x1, y1, x2, y2)
    
    # 进行分割
    segments_with_bbox = segmenter.process(image, bbox=bbox)
    print(f"边界框提示分割结果数量: {len(segments_with_bbox)}")
    
    # 可视化结果
    vis_image_bbox = visualizer.draw_segments(image.copy(), segments_with_bbox)
    
    # 绘制边界框
    cv2.rectangle(vis_image_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # 6. 示例 3: 自动分割（无提示）
    print("示例 3: 自动分割（无提示）")
    
    # 进行自动分割
    segments_auto = segmenter.process(image)
    print(f"自动分割结果数量: {len(segments_auto)}")
    
    # 可视化结果
    vis_image_auto = visualizer.draw_segments(image.copy(), segments_auto)

    # 7. 显示结果
    cv2.imshow("Segmentation with Points", vis_image_points)
    cv2.imshow("Segmentation with BBox", vis_image_bbox)
    cv2.imshow("Auto Segmentation", vis_image_auto)
    
    print("按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
