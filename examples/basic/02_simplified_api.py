"""
02_simplified_api.py
简化API使用示例

这个示例展示了如何使用VisionFramework的简化API进行快速开发，包括：
1. 使用process_image()静态方法一行代码处理图像
2. 使用run_video()静态方法一行代码处理视频
3. 使用类方法快速创建配置好的管道

用法:
  python examples/02_simplified_api.py
"""
import os
import cv2
import numpy as np
from pathlib import Path

# 修复OpenMP库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from visionframework import VisionPipeline, Visualizer


def main():
    print("=== 简化API使用示例 ===")
    
    # 创建测试图像
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (128, 128, 128)  # 灰色背景
    
    # 在图像上绘制一些简单的形状
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.circle(frame, (320, 240), 50, (0, 255, 0), -1)
    
    print("\n1. 使用process_image()静态方法处理图像...")
    # 一行代码处理图像，使用默认模型
    results1 = VisionPipeline.process_image(frame)
    detections1 = results1.get("detections", [])
    print(f"   检测结果: {len(detections1)} 个目标")
    
    print("\n2. 使用process_image()静态方法，自定义模型和参数...")
    # 一行代码处理图像，使用自定义模型和参数
    results2 = VisionPipeline.process_image(
        frame,
        model_path="yolov8n.pt",
        enable_tracking=False,
        conf_threshold=0.3
    )
    detections2 = results2.get("detections", [])
    tracks2 = results2.get("tracks", [])
    print(f"   检测结果: {len(detections2)} 个目标，{len(tracks2)} 个跟踪")
    
    print("\n3. 使用类方法创建带跟踪的管道...")
    # 使用with_tracking()类方法快速创建带跟踪的管道
    pipeline = VisionPipeline.with_tracking()
    pipeline.initialize()
    results3 = pipeline.process(frame)
    detections3 = results3.get("detections", [])
    tracks3 = results3.get("tracks", [])
    print(f"   检测结果: {len(detections3)} 个目标，{len(tracks3)} 个跟踪")
    
    print("\n4. 使用from_model()类方法创建自定义管道...")
    # 使用from_model()类方法快速创建自定义配置的管道
    pipeline = VisionPipeline.from_model(
        model_path="yolov8n.pt",
        enable_tracking=True,
        conf_threshold=0.25
    )
    pipeline.initialize()
    results4 = pipeline.process(frame)
    detections4 = results4.get("detections", [])
    tracks4 = results4.get("tracks", [])
    print(f"   检测结果: {len(detections4)} 个目标，{len(tracks4)} 个跟踪")
    
    print("\n5. 使用with_tracking()类方法并自定义检测器配置...")
    # 使用with_tracking()类方法并自定义检测器配置
    pipeline = VisionPipeline.with_tracking({
        "detector_config": {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.2
        }
    })
    pipeline.initialize()
    results5 = pipeline.process(frame)
    detections5 = results5.get("detections", [])
    tracks5 = results5.get("tracks", [])
    print(f"   检测结果: {len(detections5)} 个目标，{len(tracks5)} 个跟踪")
    
    # 可视化最后一次的检测结果
    viz = Visualizer()
    vis_frame = viz.draw_detections(frame, detections5)
    vis_frame = viz.draw_tracks(vis_frame, tracks5, draw_history=True)
    
    output_path = "output_simplified_api.jpg"
    cv2.imwrite(output_path, vis_frame)
    print(f"\n可视化结果已保存到: {output_path}")
    
    print("\n=== 示例完成 ===")
    print("简化API使开发更加便捷，适合快速原型开发和简单应用场景。")
    print("你可以根据需要选择适合的API风格：")
    print("  - 静态方法：适合简单应用，一行代码完成任务")
    print("  - 类方法：适合需要自定义配置但仍希望快速开发的场景")
    print("  - 完整配置：适合需要精细控制的复杂应用")


if __name__ == "__main__":
    main()