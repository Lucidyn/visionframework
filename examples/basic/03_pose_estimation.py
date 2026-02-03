"""
03_pose_estimation.py

基础姿态估计示例：
- 使用 PoseEstimator 对单张图片进行人体姿态估计
- 可视化姿态估计结果

注意：
- 需要提前准备姿态估计模型权重（如 yolov8n-pose.pt），并放在当前工作目录或指定路径。
"""

import cv2

from visionframework import PoseEstimator, Visualizer


def main() -> None:
    # 1. 创建姿态估计器配置
    pose_config = {
        "model_path": "yolov8n-pose.pt",  # 可替换为你的模型路径
        "device": "auto",
        "conf_threshold": 0.25,
    }

    pose_estimator = PoseEstimator(pose_config)

    # 2. 初始化模型（加载权重）
    if not pose_estimator.initialize():
        print("PoseEstimator 初始化失败，请检查模型路径和依赖（ultralytics、torch 等）。")
        return

    # 3. 读取测试图片
    image_path = "test.jpg"  # 请替换为你自己的图片路径
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    # 4. 进行姿态估计
    poses = pose_estimator.process(image)
    print(f"检测到 {len(poses)} 个人体姿态")

    # 5. 使用集成的可视化 API 绘制姿态估计结果
    visualizer = Visualizer()
    vis_image = visualizer.draw_poses(image.copy(), poses)

    # 6. 显示 / 保存结果
    cv2.imshow("Pose Estimation", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
