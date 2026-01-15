"""
CLIP 零样本分类示例

本示例展示如何使用 CLIP 进行零样本图像分类，无需训练即可识别任意类别。
CLIP (Contrastive Language-Image Pre-training) 是 OpenAI 开发的视觉-语言预训练模型。

本示例包含：
- 基本的零样本分类
- 批量图像分类
- 与检测结果的结合
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from visionframework import CLIPExtractor, Detector, Visualizer
from PIL import Image


def example_basic_zero_shot_classification():
    """
    示例 1: 基本零样本分类
    
    本示例展示如何使用 CLIP 进行基本的零样本图像分类。
    """
    print("=" * 70)
    print("示例 1: CLIP 基本零样本分类")
    print("=" * 70)
    
    # ========== 步骤 1: 初始化 CLIP 提取器 ==========
    print("\n1. 初始化 CLIP 模型...")
    
    clip = CLIPExtractor({
        "device": "cpu",              # 设备类型：可选 "cpu", "cuda"
        "use_fp16": False,            # FP16 加速（仅在 CUDA 上有效）
        "model_name": "openai/clip-vit-base-patch32"  # 模型名称
    })
    
    if not clip.initialize():
        print("✗ CLIP 初始化失败！")
        print("  提示: 请确保已安装依赖: pip install transformers pillow")
        return
    
    print("  ✓ CLIP 模型初始化成功")
    
    # ========== 步骤 2: 创建测试图像 ==========
    print("\n2. 创建测试图像...")
    
    # 创建一个简单的彩色图像（可以替换为真实图像路径）
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    print("  ✓ 测试图像已创建")
    
    # ========== 步骤 3: 定义分类标签 ==========
    print("\n3. 定义分类标签...")
    
    # CLIP 可以进行任意文本的分类
    labels = [
        "a photo of a cat",
        "a photo of a dog",
        "a person",
        "a car",
        "a tree"
    ]
    print(f"  ✓ 定义了 {len(labels)} 个分类标签")
    
    # ========== 步骤 4: 运行零样本分类 ==========
    print("\n4. 运行零样本分类...")
    
    # zero_shot_classify() 返回每个标签的置信度分数
    scores = clip.zero_shot_classify(img, labels)
    
    print("  ✓ 分类完成\n")
    
    # ========== 步骤 5: 显示分类结果 ==========
    print("分类结果（按置信度排序）:")
    
    # 将标签和分数配对，然后按分数排序
    results = list(zip(labels, scores))
    results.sort(key=lambda x: x[1], reverse=True)
    
    for i, (label, score) in enumerate(results, 1):
        print(f"  {i}. {label:30s} - {score:.4f}")


def example_batch_classification():
    """
    示例 2: 批量图像分类
    
    本示例展示如何批量分类多张图像。
    """
    print("\n" + "=" * 70)
    print("示例 2: CLIP 批量图像分类")
    print("=" * 70)
    
    # ========== 步骤 1: 初始化 CLIP ==========
    print("\n1. 初始化 CLIP 模型...")
    
    clip = CLIPExtractor({
        "device": "cpu",
        "use_fp16": False
    })
    
    if not clip.initialize():
        print("✗ CLIP 初始化失败！")
        return
    
    print("  ✓ CLIP 模型初始化成功")
    
    # ========== 步骤 2: 创建多个测试图像 ==========
    print("\n2. 创建批量测试图像...")
    
    # 创建不同颜色的图像
    images = [
        Image.new("RGB", (224, 224), color=(255, 0, 0)),    # 红色
        Image.new("RGB", (224, 224), color=(0, 255, 0)),    # 绿色
        Image.new("RGB", (224, 224), color=(0, 0, 255)),    # 蓝色
    ]
    print(f"  ✓ 创建了 {len(images)} 张测试图像")
    
    # ========== 步骤 3: 定义分类标签 ==========
    labels = ["red", "green", "blue", "yellow", "gray"]
    
    # ========== 步骤 4: 批量分类 ==========
    print("\n3. 运行批量分类...")
    
    for i, img in enumerate(images):
        scores = clip.zero_shot_classify(img, labels)
        # 找到最高分的标签
        best_label = labels[scores.index(max(scores))]
        best_score = max(scores)
        print(f"  图像 {i+1}: {best_label} (置信度: {best_score:.4f})")


def example_detection_with_clip():
    """
    示例 3: 将 CLIP 与检测结果结合
    
    本示例展示如何先进行目标检测，然后对检测到的对象使用 CLIP 进行分类。
    """
    print("\n" + "=" * 70)
    print("示例 3: CLIP 与目标检测的结合")
    print("=" * 70)
    
    # ========== 步骤 1: 初始化检测器 ==========
    print("\n1. 初始化检测器...")
    
    detector = Detector({
        "model_path": "yolov8n.pt",
        "conf_threshold": 0.25,
        "device": "cpu"
    })
    
    if not detector.initialize():
        print("✗ 检测器初始化失败！")
        return
    
    print("  ✓ 检测器初始化成功")
    
    # ========== 步骤 2: 初始化 CLIP ==========
    print("\n2. 初始化 CLIP 模型...")
    
    clip = CLIPExtractor({
        "device": "cpu",
        "use_fp16": False
    })
    
    if not clip.initialize():
        print("✗ CLIP 初始化失败！")
        return
    
    print("  ✓ CLIP 模型初始化成功")
    
    # ========== 步骤 3: 创建测试图像 ==========
    print("\n3. 创建测试图像...")
    
    # 创建一个简单的测试图像
    test_image = np.zeros((640, 480, 3), dtype=np.uint8)
    test_image[:] = (100, 100, 100)
    
    # 添加一些彩色矩形
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), -1)
    cv2.rectangle(test_image, (300, 200), (400, 300), (255, 0, 0), -1)
    
    print("  ✓ 测试图像已创建")
    
    # ========== 步骤 4: 运行检测 ==========
    print("\n4. 运行目标检测...")
    
    detections = detector.detect(test_image)
    print(f"  ✓ 检测到 {len(detections)} 个对象")
    
    # ========== 步骤 5: 对检测结果使用 CLIP 分类 ==========
    print("\n5. 对检测结果进行 CLIP 分类...")
    
    # 定义可能的对象类别
    object_labels = ["person", "car", "dog", "cat", "tree", "building"]
    
    # 遍历每个检测结果
    for i, det in enumerate(detections):
        print(f"\n  检测对象 {i+1}:")
        print(f"    YOLO 类别: {det.class_name}")
        print(f"    YOLO 置信度: {det.confidence:.4f}")
        
        # 注意: 实际应用中应该裁剪检测区域后进行 CLIP 分类
        # 这里为了简化示例，直接使用整个图像
        # 如果要裁剪检测区域，使用:
        # bbox = det.bbox
        # cropped = test_image[int(bbox.y1):int(bbox.y2), int(bbox.x1):int(bbox.x2)]
        # img_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        
        # 使用 CLIP 进行补充分类（如果有特定的标签需要细分）
        print(f"    CLIP 可用于进一步细化分类或零样本场景")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("Vision Framework - CLIP 零样本分类示例")
    print("=" * 70)
    print("\nCLIP 是一种零样本学习模型，可以对任意文本进行图像分类")
    print("=" * 70)
    
    # 运行示例
    try:
        example_basic_zero_shot_classification()
        example_batch_classification()
        example_detection_with_clip()
        
        print("\n" + "=" * 70)
        print("所有示例运行完成！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n提示:")
    print("1. CLIP 依赖: pip install transformers pillow")
    print("2. CUDA 加速: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("3. 更多信息: https://github.com/openai/CLIP")


if __name__ == "__main__":
    main()
