"""
使用配置文件示例

本示例展示如何从 YAML 或 JSON 配置文件加载配置来初始化 Vision Framework 的各个组件。

配置文件可以让你：
- 将配置与代码分离，便于管理
- 快速切换不同的配置而无需修改代码
- 在团队中共享标准配置
- 支持多种环境（开发、测试、生产）的不同配置

使用方法：
1. 复制 config_example.yaml 为 config.yaml
2. 根据需要修改配置
3. 运行此示例：python examples/config_example.py
"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from visionframework import Detector, Tracker, VisionPipeline, Visualizer, Config, ROIDetector, Counter


def example_load_from_yaml():
    """
    示例 1: 从 YAML 配置文件加载配置
    
    这个示例展示如何从 YAML 文件加载配置并使用它来初始化检测器和跟踪器。
    """
    print("=" * 70)
    print("示例 1: 从 YAML 配置文件加载配置")
    print("=" * 70)
    
    try:
        # 步骤 1: 从 YAML 文件加载配置
        # 配置文件路径相对于项目根目录
        config_path = "config_example.yaml"
        config = Config.load_from_file(config_path)
        
        print(f"✓ 成功加载配置文件: {config_path}")
        
        # 步骤 2: 提取检测器配置
        # 配置文件结构见 config_example.yaml
        detector_config = config.get("detector", {})
        print(f"  检测器配置: model_type={detector_config.get('model_type')}, "
              f"conf_threshold={detector_config.get('conf_threshold')}")
        
        # 步骤 3: 创建并初始化检测器
        detector = Detector(detector_config)
        if detector.initialize():
            print("✓ 检测器初始化成功")
        else:
            print("✗ 检测器初始化失败（可能需要下载模型）")
            return
        
        # 步骤 4: 加载测试图像
        # 注意: 这里使用示例图像路径，实际使用时请替换为真实路径
        image_path = "path/to/your/image.jpg"
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"⚠ 无法加载图像: {image_path}")
            print("  提示: 请将 image_path 替换为实际的图像路径")
            # 创建一个测试图像用于演示
            image = cv2.imread("yolov8n.pt") if Path("yolov8n.pt").exists() else None
            if image is None:
                import numpy as np
                image = np.zeros((640, 480, 3), dtype=np.uint8)
                image[:] = (128, 128, 128)
                print("  使用测试图像继续演示...")
        
        # 步骤 5: 运行检测
        # 从配置读取 categories（可为 None、class name 列表或 id 列表）
        cfg_categories = detector_config.get('categories') if isinstance(detector_config, dict) else None
        detections = detector.detect(image, categories=cfg_categories)
        print(f"✓ 检测完成，发现 {len(detections)} 个对象")
        
        # 步骤 6: 使用配置文件中的可视化器配置
        visualizer_config = config.get("visualizer", {})
        visualizer = Visualizer(visualizer_config)
        print(f"✓ 可视化器已配置: show_labels={visualizer_config.get('show_labels')}")
        
        # 可视化结果
        result_image = visualizer.draw_detections(image, detections)
        cv2.imwrite("output_config_example.jpg", result_image)
        print("✓ 结果已保存到: output_config_example.jpg")
        
    except FileNotFoundError as e:
        print(f"✗ 配置文件未找到: {e}")
        print("  提示: 请确保 config_example.yaml 存在于项目根目录")
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("  提示: YAML 配置文件需要 PyYAML，安装: pip install pyyaml")
    except Exception as e:
        print(f"✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()


def example_pipeline_from_config():
    """
    示例 2: 从配置文件加载完整管道配置
    
    这个示例展示如何从配置文件加载完整的 VisionPipeline 配置。
    """
    print("\n" + "=" * 70)
    print("示例 2: 从配置文件加载完整管道配置")
    print("=" * 70)
    
    try:
        # 步骤 1: 加载配置文件
        config = Config.load_from_file("config_example.yaml")
        
        # 步骤 2: 提取管道配置
        pipeline_config_raw = config.get("pipeline", {})
        
        # 步骤 3: 构建管道配置
        # 可以从配置文件的不同部分组合配置
        pipeline_config = {
            "enable_tracking": pipeline_config_raw.get("enable_tracking", True),
            "detector_config": config.get("detector", {}),
            "tracker_config": config.get("tracker", {})
        }
        
        print("✓ 管道配置已构建")
        print(f"  启用跟踪: {pipeline_config['enable_tracking']}")
        
        # 步骤 4: 创建并初始化管道
        pipeline = VisionPipeline(pipeline_config)
        if pipeline.initialize():
            print("✓ 管道初始化成功")
        else:
            print("✗ 管道初始化失败")
            return
        
        # 步骤 5: 处理图像（示例）
        # 实际使用时，这里应该是视频帧或图像
        print("✓ 管道已就绪，可以使用 pipeline.process(frame) 处理图像/视频")
        
    except Exception as e:
        print(f"✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()


def example_roi_from_config():
    """
    示例 3: 从配置文件加载 ROI 配置
    
    这个示例展示如何从配置文件加载 ROI（感兴趣区域）配置。
    """
    print("\n" + "=" * 70)
    print("示例 3: 从配置文件加载 ROI 配置")
    print("=" * 70)
    
    try:
        # 步骤 1: 加载配置文件
        config = Config.load_from_file("config_example.yaml")
        
        # 步骤 2: 提取 ROI 配置
        rois_config = config.get("rois", [])
        
        if not rois_config:
            print("⚠ 配置文件中没有 ROI 配置")
            return
        
        print(f"✓ 找到 {len(rois_config)} 个 ROI 配置")
        
        # 步骤 3: 构建 ROI 检测器配置
        roi_detector_config = {"rois": rois_config}
        
        # 步骤 4: 创建 ROI 检测器
        roi_detector = ROIDetector(roi_detector_config)
        if roi_detector.initialize():
            print("✓ ROI 检测器初始化成功")
            
            # 显示 ROI 信息
            for roi in rois_config:
                print(f"  - ROI '{roi.get('name')}': 类型={roi.get('type')}, "
                      f"点数={len(roi.get('points', []))}")
        else:
            print("✗ ROI 检测器初始化失败")
        
        # 步骤 5: 创建计数器（如果需要）
        counter_config = config.get("counter", {})
        if counter_config:
            counter_config["roi_detector"] = roi_detector_config
            counter = Counter(counter_config)
            if counter.initialize():
                print("✓ 计数器初始化成功")
                print(f"  计数进入: {counter_config.get('count_entering', False)}")
                print(f"  计数退出: {counter_config.get('count_exiting', False)}")
        
    except Exception as e:
        print(f"✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()


def example_create_config_file():
    """
    示例 4: 使用代码创建配置文件
    
    这个示例展示如何使用 Config 类的方法来创建和保存配置文件。
    """
    print("\n" + "=" * 70)
    print("示例 4: 使用代码创建配置文件")
    print("=" * 70)
    
    try:
        # 步骤 1: 获取默认配置
        detector_config = Config.get_default_detector_config()
        tracker_config = Config.get_default_tracker_config()
        pipeline_config = Config.get_default_pipeline_config()
        visualizer_config = Config.get_default_visualizer_config()
        
        # 步骤 2: 自定义配置
        detector_config["model_path"] = "yolov8n.pt"
        detector_config["conf_threshold"] = 0.3
        
        # 步骤 3: 构建完整配置
        full_config = {
            "detector": detector_config,
            "tracker": tracker_config,
            "pipeline": pipeline_config,
            "visualizer": visualizer_config,
            "rois": [
                {
                    "name": "zone1",
                    "type": "rectangle",
                    "points": [(100, 100), (400, 300)]
                }
            ]
        }
        
        # 步骤 4: 保存为 JSON 文件
        json_path = "my_config.json"
        Config.save_to_file(full_config, json_path, format="json")
        print(f"✓ 配置已保存为 JSON: {json_path}")
        
        # 步骤 5: 保存为 YAML 文件（如果支持）
        try:
            yaml_path = "my_config.yaml"
            Config.save_to_file(full_config, yaml_path, format="yaml")
            print(f"✓ 配置已保存为 YAML: {yaml_path}")
        except ImportError:
            print("⚠ YAML 保存跳过（需要安装 PyYAML）")
        
        # 步骤 6: 验证加载
        loaded_config = Config.load_from_file(json_path)
        print(f"✓ 验证: 成功从 {json_path} 加载配置")
        print(f"  检测器模型: {loaded_config['detector']['model_path']}")
        
    except Exception as e:
        print(f"✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("Vision Framework - 配置文件使用示例")
    print("=" * 70)
    print("\n本示例展示如何使用 YAML/JSON 配置文件来初始化框架组件")
    print("=" * 70)
    
    # 运行所有示例
    example_load_from_yaml()
    example_pipeline_from_config()
    example_roi_from_config()
    example_create_config_file()
    
    print("\n" + "=" * 70)
    print("示例运行完成！")
    print("=" * 70)
    print("\n提示:")
    print("1. 复制 config_example.yaml 为 config.yaml 并根据需要修改")
    print("2. 使用 Config.load_from_file() 加载配置")
    print("3. 将配置传递给 Detector、Tracker、VisionPipeline 等组件")
    print("4. YAML 支持需要安装: pip install pyyaml")
    print("=" * 70)


if __name__ == "__main__":
    main()

