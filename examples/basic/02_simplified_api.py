"""
02_simplified_api.py

简化 API 使用示例：
- 使用 process_image 一行调用完成检测 / 跟踪 / 分割 / 姿态估计
"""

import cv2

from visionframework import process_image, Visualizer


def main() -> None:
    # 1. 加载图像
    image_path = "test.jpg"  # 请替换为你自己的图片路径
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    # 2. 使用简化 API 进行处理
    results = process_image(
        image,
        model_path="yolov8n.pt",
        enable_tracking=True,
        enable_segmentation=False,
        enable_pose_estimation=False,
    )

    detections = results.get("detections", [])
    tracks = results.get("tracks", [])
    poses = results.get("poses", [])
    print(f"检测到 {len(detections)} 个目标, {len(tracks)} 条轨迹, {len(poses)} 个姿态")

    # 3. 使用集成的可视化 API 绘制检测 / 跟踪 / 姿态结果
    visualizer = Visualizer()
    vis_image = visualizer.draw_results(
        image.copy(),
        detections=detections,
        tracks=tracks,
        poses=poses,
    )

    # 4. 显示结果
    cv2.imshow("Simplified API - process_image", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

