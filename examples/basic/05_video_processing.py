"""
05_video_processing.py

基础视频处理示例：
- 使用 process_video 函数处理视频文件
- 支持检测、跟踪、姿态估计等多种功能
- 可视化视频处理结果

注意：
- 需要提前准备模型权重（如 yolov8n.pt），并放在当前工作目录或指定路径。
"""

import cv2

from visionframework import process_video, Visualizer


def main() -> None:
    # 1. 配置视频处理参数
    video_path = "test.mp4"  # 请替换为你的视频路径
    output_path = "output_processed.mp4"  # 输出视频路径
    
    # 2. 定义处理回调函数，用于可视化每一帧的结果
    def process_frame_callback(frame, results):
        """处理每一帧的回调函数"""
        detections = results.get("detections", [])
        tracks = results.get("tracks", [])
        poses = results.get("poses", [])
        
        print(f"Frame: 检测到 {len(detections)} 个目标, {len(tracks)} 条轨迹, {len(poses)} 个姿态")
        
        # 可视化结果
        visualizer = Visualizer()
        vis_frame = visualizer.draw_results(
            frame.copy(),
            detections=detections,
            tracks=tracks,
            poses=poses
        )
        
        return vis_frame
    
    # 3. 使用 process_video 函数处理视频
    print(f"开始处理视频: {video_path}")
    print(f"输出视频: {output_path}")
    
    # 示例 1: 基础检测 + 跟踪
    print("示例 1: 基础检测 + 跟踪")
    
    success = process_video(
        video_path=video_path,
        output_path=output_path,
        model_path="yolov8n.pt",  # 可替换为你的模型路径
        device="auto",
        enable_tracking=True,
        enable_segmentation=False,
        enable_pose_estimation=False,
        conf_threshold=0.25,
        frame_callback=process_frame_callback,
        display=True,  # 显示处理过程
        fps=30,  # 输出视频帧率
    )
    
    if success:
        print(f"视频处理完成! 输出已保存到: {output_path}")
    else:
        print("视频处理失败，请检查视频路径和模型配置。")
    
    # 示例 2: 检测 + 跟踪 + 姿态估计
    # 注意：启用姿态估计会增加处理时间和内存使用
    """
    print("\n示例 2: 检测 + 跟踪 + 姿态估计")
    output_path_pose = "output_processed_with_pose.mp4"
    
    success_pose = process_video(
        video_path=video_path,
        output_path=output_path_pose,
        model_path="yolov8n-pose.pt",  # 姿态估计模型
        device="auto",
        enable_tracking=True,
        enable_segmentation=False,
        enable_pose_estimation=True,
        conf_threshold=0.25,
        frame_callback=process_frame_callback,
        display=False,  # 不显示处理过程，直接保存
        fps=30,
    )
    
    if success_pose:
        print(f"带姿态估计的视频处理完成! 输出已保存到: {output_path_pose}")
    else:
        print("带姿态估计的视频处理失败，请检查视频路径和模型配置。")
    """


if __name__ == "__main__":
    main()
