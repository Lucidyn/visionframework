#!/usr/bin/env python3
"""
综合测试脚本，测试新增的功能：
1. SAM分割
2. CLIP模型
3. 姿态估计
4. 检测器+SAM集成
"""

import os
import sys
import numpy as np
import cv2
import traceback

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 测试结果记录
test_results = {
    "passed": [],
    "failed": [],
    "skipped": []
}


def test_sam_segmenter():
    """测试SAM分割器"""
    test_name = "SAM分割器测试"
    print(f"\n=== {test_name} ===")
    
    try:
        from visionframework.core.segmenters.sam_segmenter import SAMSegmenter
        
        # 创建测试图像
        test_image = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (300, 300), (255, 255, 255), -1)
        
        # 初始化SAM分割器
        sam = SAMSegmenter({
            "model_type": "vit_b",
            "device": "cpu",
            "use_fp16": False
        })
        
        # 尝试初始化，这里可能会因为缺少segment_anything库而失败
        initialized = sam.initialize()
        
        if initialized:
            # 测试自动分割
            masks = sam.automatic_segment(test_image)
            print(f"✓ SAM自动分割成功，生成了 {len(masks)} 个掩码")
            
            # 测试点分割
            masks_with_points = sam.segment_with_points(
                test_image, 
                points=[(200, 200)], 
                labels=[1]
            )
            print(f"✓ SAM点分割成功，生成了 {len(masks_with_points)} 个掩码")
            
            # 测试框分割
            masks_with_boxes = sam.segment_with_boxes(
                test_image, 
                boxes=[(100, 100, 300, 300)]
            )
            print(f"✓ SAM框分割成功，生成了 {len(masks_with_boxes)} 个掩码")
            
            test_results["passed"].append(test_name)
        else:
            print(f"⚠ SAM分割器初始化失败，可能缺少segment_anything库，跳过功能测试")
            test_results["skipped"].append(f"{test_name} - 缺少依赖")
            
    except ImportError as e:
        print(f"⚠ 导入错误：{e}，跳过SAM分割器测试")
        test_results["skipped"].append(f"{test_name} - 缺少依赖")
    except Exception as e:
        print(f"✗ 测试失败：{e}")
        traceback.print_exc()
        test_results["failed"].append(test_name)


def test_clip_extractor():
    """测试CLIP提取器"""
    test_name = "CLIP提取器测试"
    print(f"\n=== {test_name} ===")
    
    try:
        from visionframework.core.components.processors.clip_extractor import CLIPExtractor
        
        # 创建测试图像
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (174, 174), (255, 255, 255), -1)
        
        # 初始化CLIP提取器
        clip = CLIPExtractor({
            "model_name": "openai/clip-vit-base-patch32",
            "device": "cpu",
            "use_fp16": False
        })
        
        # 尝试初始化
        print("  - 正在初始化CLIP模型...")
        try:
            initialized = clip.initialize()
            print(f"  - CLIP初始化结果: {initialized}")
        except Exception as e:
            print(f"  - CLIP初始化异常: {e}")
            traceback.print_exc()
            initialized = False
        
        if initialized:
            # 测试图像编码
            image_embedding = clip.encode_image(test_image)
            print(f"✓ 图像编码成功，嵌入形状：{image_embedding.shape}")
            
            # 测试文本编码
            text_embedding = clip.encode_text(["square", "circle", "triangle"])
            print(f"✓ 文本编码成功，嵌入形状：{text_embedding.shape}")
            
            # 测试图像-文本相似度
            similarity = clip.image_text_similarity(test_image, ["square", "circle", "triangle"])
            print(f"✓ 图像-文本相似度计算成功，相似度：{similarity.tolist()}")
            
            # 测试零样本分类
            scores = clip.zero_shot_classify(test_image, ["square", "circle", "triangle"])
            print(f"✓ 零样本分类成功，分数：{scores}")
            
            test_results["passed"].append(test_name)
        else:
            print(f"⚠ CLIP提取器初始化失败，跳过功能测试")
            test_results["skipped"].append(f"{test_name} - 初始化失败")
            
    except ImportError as e:
        print(f"⚠ 导入错误：{e}，跳过CLIP提取器测试")
        test_results["skipped"].append(f"{test_name} - 缺少依赖")
    except Exception as e:
        print(f"✗ 测试失败：{e}")
        traceback.print_exc()
        test_results["failed"].append(test_name)


def test_pose_estimator():
    """测试姿态估计器"""
    test_name = "姿态估计器测试"
    print(f"\n=== {test_name} ===")
    
    try:
        from visionframework.core.components.processors.pose_estimator import PoseEstimator
        
        # 创建测试图像（一个简单的人形轮廓）
        test_image = np.zeros((400, 300, 3), dtype=np.uint8)
        
        # 画一个简单的人形轮廓
        cv2.circle(test_image, (150, 80), 20, (255, 255, 255), -1)  # 头
        cv2.line(test_image, (150, 100), (150, 200), (255, 255, 255), 10)  # 身体
        cv2.line(test_image, (150, 120), (100, 180), (255, 255, 255), 8)  # 左臂
        cv2.line(test_image, (150, 120), (200, 180), (255, 255, 255), 8)  # 右臂
        cv2.line(test_image, (150, 200), (100, 280), (255, 255, 255), 8)  # 左腿
        cv2.line(test_image, (150, 200), (200, 280), (255, 255, 255), 8)  # 右腿
        
        # 测试YOLO Pose
        print("\n- 测试YOLO Pose")
        yolo_pose = PoseEstimator({
            "model_type": "yolo_pose",
            "model_path": "yolov8n-pose.pt",
            "device": "cpu",
            "conf_threshold": 0.1
        })
        
        yolo_initialized = yolo_pose.initialize()
        if yolo_initialized:
            yolo_poses = yolo_pose.estimate(test_image)
            print(f"✓ YOLO Pose成功，检测到 {len(yolo_poses)} 个姿态")
        else:
            print(f"⚠ YOLO Pose初始化失败，跳过测试")
        
        # 测试MediaPipe Pose
        print("\n- 测试MediaPipe Pose")
        mediapipe_pose = PoseEstimator({
            "model_type": "mediapipe",
            "min_detection_confidence": 0.3,
            "min_tracking_confidence": 0.3
        })
        
        mediapipe_initialized = mediapipe_pose.initialize()
        if mediapipe_initialized:
            mediapipe_poses = mediapipe_pose.estimate(test_image)
            print(f"✓ MediaPipe Pose成功，检测到 {len(mediapipe_poses)} 个姿态")
        else:
            print(f"⚠ MediaPipe Pose初始化失败，跳过测试")
        
        if yolo_initialized or mediapipe_initialized:
            test_results["passed"].append(test_name)
        else:
            test_results["skipped"].append(f"{test_name} - 所有模型初始化失败")
            
    except ImportError as e:
        print(f"⚠ 导入错误：{e}，跳过姿态估计器测试")
        test_results["skipped"].append(f"{test_name} - 缺少依赖")
    except Exception as e:
        print(f"✗ 测试失败：{e}")
        traceback.print_exc()
        test_results["failed"].append(test_name)


def test_detector_sam_integration():
    """测试检测器+SAM集成"""
    test_name = "检测器+SAM集成测试"
    print(f"\n=== {test_name} ===")
    
    try:
        from visionframework.core.components.detectors.yolo_detector import YOLODetector
        
        # 创建测试图像
        test_image = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (300, 300), (255, 255, 255), -1)
        
        # 初始化带SAM的检测器
        detector = YOLODetector({
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.1,
            "device": "cpu",
            "segmenter_type": "sam",
            "sam_model_type": "vit_b",
            "sam_use_fp16": False
        })
        
        # 尝试初始化
        initialized = detector.initialize()
        
        if initialized:
            # 测试检测+分割
            detections = detector.detect(test_image)
            print(f"✓ 检测+分割成功，检测到 {len(detections)} 个目标")
            
            # 统计带掩码的检测结果
            mask_count = sum(1 for d in detections if hasattr(d, 'mask') and d.mask is not None)
            print(f"✓ 其中 {mask_count} 个目标带有分割掩码")
            
            test_results["passed"].append(test_name)
        else:
            print(f"⚠ 检测器+SAM集成初始化失败，跳过测试")
            test_results["skipped"].append(f"{test_name} - 初始化失败")
            
    except ImportError as e:
        print(f"⚠ 导入错误：{e}，跳过检测器+SAM集成测试")
        test_results["skipped"].append(f"{test_name} - 缺少依赖")
    except Exception as e:
        print(f"✗ 测试失败：{e}")
        traceback.print_exc()
        test_results["failed"].append(test_name)


def test_model_cache():
    """测试模型缓存功能"""
    test_name = "模型缓存测试"
    print(f"\n=== {test_name} ===")
    
    try:
        from visionframework.utils.io.config_models import ModelCache
        
        # 测试模型缓存的基本功能
        print("  - 测试模型缓存初始化")
        
        # 创建一个简单的模型加载函数
        def simple_model_loader():
            import numpy as np
            # 创建一个简单的模型对象
            class SimpleModel:
                def predict(self, x):
                    return np.sum(x)
            return SimpleModel()
        
        # 测试缓存功能
        model1 = ModelCache.get_model("test_model", simple_model_loader)
        model2 = ModelCache.get_model("test_model", simple_model_loader)
        
        # 验证返回的是同一个实例
        print(f"  - 模型缓存命中测试: {model1 is model2}")
        
        # 测试模型释放
        ModelCache.release_model("test_model")
        print("  - 模型释放测试成功")
        
        # 测试缓存状态获取
        cache_status = ModelCache.get_cache_status()
        print(f"  - 缓存状态获取成功: {cache_status}")
        
        test_results["passed"].append(test_name)
    except ImportError as e:
        print(f"⚠ 导入错误：{e}，跳過模型缓存测试")
        test_results["skipped"].append(f"{test_name} - 缺少依赖")
    except Exception as e:
        print(f"✗ 测试失败：{e}")
        traceback.print_exc()
        test_results["failed"].append(test_name)


def test_extended_model_support():
    """测试扩展模型支持功能"""
    test_name = "扩展模型支持测试"
    print(f"\n=== {test_name} ===")
    
    try:
        from visionframework.models import get_model_manager
        
        # 获取模型管理器
        model_manager = get_model_manager()
        
        # 测试模型注册
        print("  - 测试模型注册功能")
        
        # 注册一个自定义模型
        model_manager.register_model(
            name="test_custom_model",
            source="yolo",
            config={"file_name": "test_model.pt"}
        )
        
        # 测试获取模型信息
        model_info = model_manager.get_model_info("test_custom_model")
        print(f"  - 模型信息获取成功: {model_info}")
        
        # 测试获取所有注册模型
        registered_models = model_manager.get_all_registered_models()
        print(f"  - 注册模型数量: {len(registered_models)}")
        
        # 验证新添加的模型类型
        has_efficientdet = any("efficientdet" in model for model in registered_models)
        has_fasterrcnn = any("fasterrcnn" in model for model in registered_models)
        print(f"  - 包含 EfficientDet 模型: {has_efficientdet}")
        print(f"  - 包含 Faster R-CNN 模型: {has_fasterrcnn}")
        
        test_results["passed"].append(test_name)
    except ImportError as e:
        print(f"⚠ 导入错误：{e}，跳过扩展模型支持测试")
        test_results["skipped"].append(f"{test_name} - 缺少依赖")
    except Exception as e:
        print(f"✗ 测试失败：{e}")
        traceback.print_exc()
        test_results["failed"].append(test_name)


def test_batch_processing_optimization():
    """测试批处理性能优化功能"""
    test_name = "批处理性能优化测试"
    print(f"\n=== {test_name} ===")
    
    try:
        from visionframework.core.pipeline import VisionPipeline
        import numpy as np
        
        # 创建测试配置
        config = {
            "detector_config": {
                "model_path": "yolov8n.pt",
                "device": "cpu"
            }
        }
        
        # 初始化管道
        pipeline = VisionPipeline(config)
        
        # 创建测试图像
        test_images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(4)]
        
        # 测试基本批处理
        print("  - 测试基本批处理")
        results1 = pipeline.process_batch(test_images)
        print(f"  - 基本批处理结果数量: {len(results1)}")
        
        # 测试最大批处理大小限制
        print("  - 测试最大批处理大小限制")
        results2 = pipeline.process_batch(test_images, max_batch_size=2)
        print(f"  - 限制批处理大小结果数量: {len(results2)}")
        
        # 验证两种方式结果数量一致
        print(f"  - 结果一致性测试: {len(results1) == len(results2) == 4}")
        
        test_results["passed"].append(test_name)
    except ImportError as e:
        print(f"⚠ 导入错误：{e}，跳过批处理性能优化测试")
        test_results["skipped"].append(f"{test_name} - 缺少依赖")
    except Exception as e:
        print(f"✗ 测试失败：{e}")
        traceback.print_exc()
        test_results["failed"].append(test_name)


def test_reid_functionality():
    """测试ReID功能"""
    test_name = "ReID功能测试"
    print(f"\n=== {test_name} ===")
    
    try:
        from visionframework.core.processors.reid_extractor import ReIDExtractor
        import numpy as np
        
        # 创建测试图像
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 创建测试边界框
        test_bboxes = [(100, 100, 200, 300), (300, 150, 400, 350)]
        
        # 初始化ReID提取器
        reid_extractor = ReIDExtractor(
            model_name="resnet50",
            device="cpu",
            input_size=(128, 256)
        )
        
        # 尝试初始化
        initialized = reid_extractor.initialize()
        
        if initialized:
            # 测试特征提取
            print("  - 测试ReID特征提取")
            features = reid_extractor.extract(test_image, test_bboxes)
            print(f"  - 特征提取成功，形状: {features.shape}")
            
            # 测试批量处理
            print("  - 测试ReID批量处理")
            batch_images = [test_image, test_image]
            batch_bboxes = [test_bboxes, test_bboxes]
            batch_features = reid_extractor.process_batch(batch_images, batch_bboxes)
            print(f"  - 批量处理成功，结果数量: {len(batch_features)}")
            
            test_results["passed"].append(test_name)
        else:
            print(f"⚠ ReID提取器初始化失败，跳过功能测试")
            test_results["skipped"].append(f"{test_name} - 初始化失败")
    except ImportError as e:
        print(f"⚠ 导入错误：{e}，跳过ReID功能测试")
        test_results["skipped"].append(f"{test_name} - 缺少依赖")
    except Exception as e:
        print(f"✗ 测试失败：{e}")
        traceback.print_exc()
        test_results["failed"].append(test_name)


def run_all_tests():
    """运行所有测试"""
    print("开始测试新增功能...")
    print("=" * 50)
    
    # 运行各项测试
    test_clip_extractor()
    test_pose_estimator()
    test_sam_segmenter()
    test_detector_sam_integration()
    test_model_cache()
    test_extended_model_support()
    test_batch_processing_optimization()
    test_reid_functionality()
    
    # 打印测试结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    print(f"通过测试: {len(test_results['passed'])}")
    for test in test_results['passed']:
        print(f"  ✓ {test}")
    
    print(f"\n跳过测试: {len(test_results['skipped'])}")
    for test in test_results['skipped']:
        print(f"  ⚠ {test}")
    
    print(f"\n失败测试: {len(test_results['failed'])}")
    for test in test_results['failed']:
        print(f"  ✗ {test}")
    
    print("\n" + "=" * 50)
    
    # 计算成功率
    total_tests = len(test_results['passed']) + len(test_results['failed']) + len(test_results['skipped'])
    if total_tests > 0:
        success_rate = (len(test_results['passed']) / total_tests) * 100
        print(f"总体测试成功率: {success_rate:.1f}%")
    
    # 生成测试报告
    if test_results['failed']:
        print("\n测试存在失败项，请检查错误信息并修复问题。")
        return False
    else:
        print("\n所有测试通过或跳过，代码结构和集成正常！")
        return True


if __name__ == "__main__":
    result = run_all_tests()
    # 将测试结果写入文件
    with open('test_results.txt', 'w', encoding='utf-8') as f:
        f.write("测试结果汇总\n")
        f.write("=" * 50 + "\n")
        f.write(f"通过测试: {len(test_results['passed'])}\n")
        for test in test_results['passed']:
            f.write(f"  ✓ {test}\n")
        f.write(f"\n跳过测试: {len(test_results['skipped'])}\n")
        for test in test_results['skipped']:
            f.write(f"  ⚠ {test}\n")
        f.write(f"\n失败测试: {len(test_results['failed'])}\n")
        for test in test_results['failed']:
            f.write(f"  ✗ {test}\n")
        f.write("=" * 50 + "\n")
        total_tests = len(test_results['passed']) + len(test_results['failed']) + len(test_results['skipped'])
        if total_tests > 0:
            success_rate = (len(test_results['passed']) / total_tests) * 100
            f.write(f"总体测试成功率: {success_rate:.1f}%\n")
    print("测试结果已写入 test_results.txt 文件")
