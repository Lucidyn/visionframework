"""
00_basic_detection.py
基础目标检测示例

这个示例展示了如何使用VisionFramework进行基础的目标检测，包括：
1. 创建VisionPipeline实例
2. 初始化检测器
3. 处理图像
4. 可视化检测结果

用法:
  python examples/00_basic_detection.py
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 修复OpenMP库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parents[1]))

from visionframework import VisionPipeline, Visualizer


def main():
    print("=== 基础目标检测示例 ===")
    
    # 创建一个空白图像作为测试
    # 实际使用时，可以替换为 cv2.imread("your_image.jpg")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (128, 128, 128)  # 灰色背景
    
    # 在图像上绘制一些简单的形状作为检测目标
    # 画一个蓝色矩形
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), -1)
    # 画一个绿色圆形
    cv2.circle(frame, (320, 240), 50, (0, 255, 0), -1)
    # 画一个红色三角形
    pts = np.array([[500, 100], [550, 200], [450, 200]], np.int32)
    cv2.fillPoly(frame, [pts], (0, 0, 255))
    
    print("创建VisionPipeline实例...")
    # 创建VisionPipeline实例，使用默认的YOLOv8n模型
    config = {
        "enable_tracking": False,  # 关闭跟踪，只进行检测
        "detector_config": {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.25  # 设置置信度阈值
        }
    }
    
    pipeline = VisionPipeline(config)
    
    print("初始化检测器...")
    # 初始化检测器
    if not pipeline.initialize():
        print("初始化失败！")
        return
    
    print("处理图像...")
    # 处理图像
    results = pipeline.process(frame)
    
    # 获取检测结果
    detections = results.get("detections", [])
    tracks = results.get("tracks", [])
    
    print(f"检测结果: {len(detections)} 个目标，{len(tracks)} 个跟踪")
    
    # 可视化检测结果
    print("可视化检测结果...")
    viz = Visualizer()
    vis_frame = viz.draw_detections(frame, detections)
    
    # 保存可视化结果
    output_path = "output_basic_detection.jpg"
    cv2.imwrite(output_path, vis_frame)
    print(f"可视化结果已保存到: {output_path}")
    
    # 显示结果（可选）
    # cv2.imshow("检测结果", vis_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    print("\n=== 示例完成 ===")
    print("你可以修改代码中的图像路径，使用自己的图像进行测试。")


if __name__ == "__main__":
    import sys
    main()