"""
YOLO Pose 姿态估计示例

本示例展示如何使用 YOLO Pose 模型进行人体姿态估计。
包括关键点检测、骨架绘制、姿态分析等功能。
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from visionframework import PoseEstimator, Visualizer


def example_basic_pose_estimation():
    """基本姿态估计示例"""
    print("=" * 70)
    print("YOLO Pose 基本姿态估计示例")
    print("=" * 70)
    
    # 初始化姿态估计器
    print("\n1. 初始化 YOLO Pose 估计器...")
    pose_estimator = PoseEstimator({
        "model_path": "yolov8n-pose.pt",  # 会自动下载
        "model_type": "yolo_pose",
        "conf_threshold": 0.25,           # 检测置信度阈值
        "keypoint_threshold": 0.5,        # 关键点置信度阈值
        "device": "cpu"                   # 可选: "cpu", "cuda", "mps"
    })
    
    if not pose_estimator.initialize():
        print("姿态估计器初始化失败！")
        print("请确保已安装 ultralytics: pip install ultralytics")
        return
    
    print("✓ YOLO Pose 估计器初始化成功")
    
    # 创建测试图像（或加载真实图像）
    print("\n2. 准备测试图像...")
    # 创建一个简单的测试图像
    test_image = np.zeros((640, 480, 3), dtype=np.uint8)
    test_image[:] = (50, 50, 50)  # 深灰色背景
    
    # 添加一些形状模拟人体
    # 头部
    cv2.circle(test_image, (320, 100), 30, (255, 255, 255), -1)
    # 身体
    cv2.rectangle(test_image, (300, 130), (340, 350), (200, 200, 200), -1)
    # 手臂
    cv2.line(test_image, (300, 180), (250, 250), (200, 200, 200), 15)
    cv2.line(test_image, (340, 180), (390, 250), (200, 200, 200), 15)
    # 腿部
    cv2.line(test_image, (310, 350), (310, 450), (200, 200, 200), 15)
    cv2.line(test_image, (330, 350), (330, 450), (200, 200, 200), 15)
    
    print("✓ 测试图像已创建")
    
    # 运行姿态估计
    print("\n3. 运行姿态估计...")
    poses = pose_estimator.estimate(test_image)
    print(f"✓ 检测到 {len(poses)} 个人体姿态")
    
    # 显示姿态信息
    for i, pose in enumerate(poses):
        print(f"\n  姿态 {i+1}:")
        print(f"    边界框: {pose.bbox}")
        print(f"    置信度: {pose.confidence:.2f}")
        print(f"    关键点数量: {len(pose.keypoints)}")
        
        # 显示部分关键点
        print("    关键点:")
        for kp in pose.keypoints[:5]:  # 显示前5个关键点
            print(f"      {kp.keypoint_name}: ({kp.x:.1f}, {kp.y:.1f}), 置信度: {kp.confidence:.2f}")
    
    # 可视化结果
    print("\n4. 可视化结果...")
    visualizer = Visualizer()
    result_image = visualizer.draw_poses(
        test_image,
        poses,
        draw_skeleton=True,      # 绘制骨架
        draw_keypoints=True,     # 绘制关键点
        draw_bbox=True          # 绘制边界框
    )
    
    # 保存结果
    output_path = "yolo_pose_output.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"✓ 结果已保存到: {output_path}")
    
    # 尝试显示图像
    try:
        cv2.imshow("YOLO Pose Estimation Result", result_image)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("无法显示图像窗口（可能在没有 GUI 的环境中运行）")


def example_pose_with_real_image(image_path):
    """使用真实图像进行姿态估计"""
    print("=" * 70)
    print("YOLO Pose 真实图像姿态估计示例")
    print("=" * 70)
    
    # 加载图像
    print(f"\n1. 加载图像: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        print("请提供有效的图像路径")
        return
    
    print(f"✓ 图像尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 初始化姿态估计器
    print("\n2. 初始化 YOLO Pose 估计器...")
    pose_estimator = PoseEstimator({
        "model_path": "yolov8n-pose.pt",
        "conf_threshold": 0.25,
        "keypoint_threshold": 0.5,
        "device": "cpu"
    })
    
    if not pose_estimator.initialize():
        print("姿态估计器初始化失败！")
        return
    
    print("✓ 姿态估计器初始化成功")
    
    # 运行姿态估计
    print("\n3. 运行姿态估计...")
    poses = pose_estimator.estimate(image)
    print(f"✓ 检测到 {len(poses)} 个人体姿态")
    
    # 按置信度排序
    poses.sort(key=lambda x: x.confidence, reverse=True)
    
    # 显示前3个姿态的详细信息
    print("\n前3个姿态的详细信息:")
    for i, pose in enumerate(poses[:3]):
        print(f"\n  姿态 {i+1} (置信度: {pose.confidence:.2f}):")
        print(f"    边界框: {pose.bbox}")
        print(f"    关键点数量: {len(pose.keypoints)}")
        
        # 显示所有关键点
        visible_keypoints = [kp for kp in pose.keypoints if kp.confidence > 0.5]
        print(f"    可见关键点: {len(visible_keypoints)}/{len(pose.keypoints)}")
        
        # 显示关键点名称
        keypoint_names = [kp.keypoint_name for kp in visible_keypoints]
        print(f"    关键点: {', '.join(keypoint_names[:10])}...")  # 显示前10个
    
    # 可视化
    print("\n4. 可视化结果...")
    visualizer = Visualizer()
    result_image = visualizer.draw_poses(
        image,
        poses,
        draw_skeleton=True,
        draw_keypoints=True,
        draw_bbox=True
    )
    
    # 添加统计信息
    info_text = f"Poses: {len(poses)}"
    cv2.putText(
        result_image,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    # 保存结果
    output_path = "yolo_pose_real_output.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"✓ 结果已保存到: {output_path}")


def example_pose_video_processing():
    """视频姿态估计示例"""
    print("=" * 70)
    print("YOLO Pose 视频处理示例")
    print("=" * 70)
    
    # 初始化姿态估计器
    print("\n1. 初始化 YOLO Pose 估计器...")
    pose_estimator = PoseEstimator({
        "model_path": "yolov8n-pose.pt",
        "conf_threshold": 0.25,
        "keypoint_threshold": 0.5,
        "device": "cpu"
    })
    
    if not pose_estimator.initialize():
        print("姿态估计器初始化失败！")
        return
    
    print("✓ 姿态估计器初始化成功")
    
    # 初始化可视化器
    visualizer = Visualizer()
    
    # 打开视频或摄像头
    print("\n2. 打开视频源...")
    print("提示: 可以修改 video_path 为视频文件路径，或使用 0 使用摄像头")
    
    video_path = 0  # 0 表示使用摄像头，或改为视频文件路径
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频源: {video_path}")
        print("使用测试图像代替...")
        
        # 创建测试图像
        test_frame = np.zeros((640, 480, 3), dtype=np.uint8)
        test_frame[:] = (50, 50, 50)
        cv2.rectangle(test_frame, (200, 100), (440, 400), (200, 200, 200), -1)
        
        # 处理测试图像
        poses = pose_estimator.estimate(test_frame)
        
        # 可视化
        result_frame = visualizer.draw_poses(
            test_frame,
            poses,
            draw_skeleton=True,
            draw_keypoints=True
        )
        
        # 保存结果
        cv2.imwrite("yolo_pose_video_test.jpg", result_frame)
        print("✓ 测试结果已保存到: yolo_pose_video_test.jpg")
        return
    
    print("处理视频中... (按 'q' 退出)")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 估计姿态
        poses = pose_estimator.estimate(frame)
        
        # 可视化
        result_frame = visualizer.draw_poses(
            frame,
            poses,
            draw_skeleton=True,
            draw_keypoints=True,
            draw_bbox=True
        )
        
        # 添加信息
        info_text = f"Frame: {frame_count} | Poses: {len(poses)}"
        cv2.putText(
            result_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # 显示每个姿态的置信度
        for i, pose in enumerate(poses):
            x1, y1, x2, y2 = map(int, pose.bbox)
            conf_text = f"Pose {i+1}: {pose.confidence:.2f}"
            cv2.putText(
                result_frame,
                conf_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )
        
        # 显示
        cv2.imshow("YOLO Pose Estimation", result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  已处理 {frame_count} 帧，检测到 {len(poses)} 个姿态")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ 处理完成，共处理 {frame_count} 帧")


def example_pose_keypoint_analysis():
    """关键点分析示例"""
    print("=" * 70)
    print("YOLO Pose 关键点分析示例")
    print("=" * 70)
    
    # 初始化姿态估计器
    print("\n1. 初始化姿态估计器...")
    pose_estimator = PoseEstimator({
        "model_path": "yolov8n-pose.pt",
        "conf_threshold": 0.25,
        "keypoint_threshold": 0.5
    })
    
    if not pose_estimator.initialize():
        print("姿态估计器初始化失败！")
        return
    
    print("✓ 姿态估计器初始化成功")
    
    # 创建测试图像
    print("\n2. 创建测试图像...")
    test_image = np.zeros((640, 480, 3), dtype=np.uint8)
    test_image[:] = (50, 50, 50)
    cv2.rectangle(test_image, (200, 100), (440, 400), (200, 200, 200), -1)
    
    # 运行姿态估计
    print("\n3. 运行姿态估计...")
    poses = pose_estimator.estimate(test_image)
    
    if len(poses) == 0:
        print("未检测到姿态，使用模拟数据进行演示...")
        print("\n关键点分析功能演示:")
        print("  - 可以计算关键点之间的距离")
        print("  - 可以分析姿态角度（如手臂角度、腿部角度）")
        print("  - 可以检测特定动作（如举手、弯腰等）")
        return
    
    print(f"✓ 检测到 {len(poses)} 个姿态")
    
    # 分析每个姿态的关键点
    print("\n4. 关键点分析...")
    for i, pose in enumerate(poses):
        print(f"\n姿态 {i+1} 的关键点分析:")
        
        # 获取特定关键点
        nose = pose.get_keypoint_by_name("nose")
        left_shoulder = pose.get_keypoint_by_name("left_shoulder")
        right_shoulder = pose.get_keypoint_by_name("right_shoulder")
        left_elbow = pose.get_keypoint_by_name("left_elbow")
        left_wrist = pose.get_keypoint_by_name("left_wrist")
        
        # 计算肩膀宽度
        if left_shoulder and right_shoulder:
            shoulder_width = ((left_shoulder.x - right_shoulder.x)**2 + 
                            (left_shoulder.y - right_shoulder.y)**2)**0.5
            print(f"  肩膀宽度: {shoulder_width:.1f} 像素")
        
        # 计算手臂长度
        if left_shoulder and left_elbow and left_wrist:
            upper_arm = ((left_shoulder.x - left_elbow.x)**2 + 
                        (left_shoulder.y - left_elbow.y)**2)**0.5
            forearm = ((left_elbow.x - left_wrist.x)**2 + 
                      (left_elbow.y - left_wrist.y)**2)**0.5
            arm_length = upper_arm + forearm
            print(f"  左臂长度: {arm_length:.1f} 像素 (上臂: {upper_arm:.1f}, 前臂: {forearm:.1f})")
        
        # 统计可见关键点
        visible_keypoints = [kp for kp in pose.keypoints if kp.confidence > 0.5]
        print(f"  可见关键点: {len(visible_keypoints)}/{len(pose.keypoints)}")
        
        # 列出所有关键点及其位置
        print("  关键点位置:")
        for kp in pose.keypoints[:10]:  # 显示前10个
            if kp.confidence > 0.5:
                print(f"    {kp.keypoint_name}: ({kp.x:.1f}, {kp.y:.1f}), 置信度: {kp.confidence:.2f}")
    
    # 可视化
    print("\n5. 可视化结果...")
    visualizer = Visualizer()
    result_image = visualizer.draw_poses(
        test_image,
        poses,
        draw_skeleton=True,
        draw_keypoints=True
    )
    
    # 保存结果
    output_path = "yolo_pose_analysis_output.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"✓ 结果已保存到: {output_path}")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("YOLO Pose 姿态估计示例")
    print("=" * 70)
    print("\n本示例展示如何使用 YOLO Pose 模型进行人体姿态估计")
    print("=" * 70)
    
    import sys
    
    # 运行示例
    try:
        # 示例 1: 基本姿态估计
        example_basic_pose_estimation()
        
        print("\n" + "-" * 70)
        
        # 示例 2: 关键点分析
        example_pose_keypoint_analysis()
        
        print("\n" + "-" * 70)
        
        # 示例 3: 如果提供了图像路径，运行真实图像示例
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            example_pose_with_real_image(image_path)
        else:
            print("\n提示: 可以传入图像路径作为参数来测试真实图像")
            print("例如: python yolo_pose_example.py your_image.jpg")
        
        print("\n" + "-" * 70)
        print("\n提示: 可以取消注释下面的代码来运行视频处理示例")
        print("# example_pose_video_processing()")
        
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

