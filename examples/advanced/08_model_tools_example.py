"""
Example script demonstrating new model tools:
- Model optimization (quantization, pruning, distillation config)
- Fine-tuning configuration
- Data augmentation
- Trajectory analysis

This example is intentionally lightweight and uses tiny toy networks instead
of real detection models so it can run quickly on CPU.
"""

import time
from typing import List
import os

# 解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import torch

from visionframework import (
    Track,
    TrajectoryAnalyzer,
    ImageAugmenter, AugmentationConfig, AugmentationType,
    QuantizationConfig, PruningConfig, quantize_model, prune_model,
    FineTuningConfig,
    select_model,
    fuse_features
)


class TinyClassifier(torch.nn.Module):
    """Very small classifier used only for demonstration."""

    def __init__(self, in_dim: int = 16, hidden: int = 8, num_classes: int = 3) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - demo only
        return self.fc2(self.relu(self.fc1(x)))


def demo_model_optimization() -> None:
    print("=== 模型优化示例（量化 & 剪枝） ===")
    model = TinyClassifier().eval()
    x = torch.randn(4, 16)

    with torch.no_grad():
        baseline_out = model(x)
    print("原始模型输出形状:", tuple(baseline_out.shape))

    # 量化
    q_config = QuantizationConfig(
        quantization_type="dynamic",
        backend="fbgemm",
        verbose=False,
    )
    try:
        q_model = quantize_model(model, q_config)
        with torch.no_grad():
            q_out = q_model(x)
        print("量化模型输出形状:", tuple(q_out.shape))
    except (RuntimeError, ValueError) as e:
        print(f"[警告] 量化后端不可用，跳过量化示例: {e}")

    # 剪枝
    p_config = PruningConfig(
        pruning_type="l1_unstructured",
        amount=0.3,
        target_modules=[torch.nn.Linear],
        global_pruning=False,
        verbose=False,
    )
    pruned_model = prune_model(model, p_config)
    with torch.no_grad():
        pruned_out = pruned_model(x)
    print("剪枝后模型输出形状:", tuple(pruned_out.shape))


def demo_fine_tuning_config() -> None:
    print("\n=== 微调配置示例（FineTuningConfig） ===")
    cfg = FineTuningConfig(
        strategy="freeze",
        epochs=5,
        batch_size=16,
        learning_rate=1e-4,
    )
    print("微调配置:")
    print(f"  策略: {cfg.strategy}")
    print(f"  轮数: {cfg.epochs}")
    print(f"  batch_size: {cfg.batch_size}")
    print(f"  学习率: {cfg.learning_rate}")
    print("（实际微调请参考训练脚本，这里只展示配置用法）")


def demo_data_augmentation() -> None:
    print("\n=== 数据增强示例（ImageAugmenter） ===")

    # 创建简单的测试图像：灰色背景 + 白色方块
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    cv2.rectangle(img, (32, 32), (96, 96), (255, 255, 255), -1)

    config = AugmentationConfig(
        augmentations=[
            AugmentationType.FLIP,
            AugmentationType.ROTATE,
            AugmentationType.BRIGHTNESS,
            AugmentationType.CONTRAST,
        ],
        probability=0.5,
    )
    augmenter = ImageAugmenter(config)

    # 在调用 augment 方法时传递增强参数
    aug_img = augmenter.augment(
        img,
        angle=15,  # 旋转角度
        brightness_factor=1.0,  # 亮度因子
        contrast_factor=1.0  # 对比度因子
    )
    print("原始图像形状:", img.shape, "增强后图像形状:", aug_img.shape)

    # 可选：保存到本地查看效果
    cv2.imwrite("aug_example_input.jpg", img)
    cv2.imwrite("aug_example_output.jpg", aug_img)
    print("已将增强前后图像保存为 aug_example_input.jpg / aug_example_output.jpg")


def demo_trajectory_analysis() -> None:
    print("\n=== 轨迹分析示例（TrajectoryAnalyzer） ===")

    analyzer = TrajectoryAnalyzer(fps=30.0, pixel_to_meter=0.1)

    # 构造一个简单的水平运动轨迹
    track = Track(
        track_id=1,
        bbox=(0, 0, 20, 20),
        confidence=1.0,
        class_id=0,
        class_name="object",
    )
    # 模拟几帧运动
    boxes: List[tuple] = [
        (0, 0, 20, 20),
        (5, 0, 25, 20),
        (10, 0, 30, 20),
        (15, 0, 35, 20),
    ]
    for bbox in boxes[1:]:
        time.sleep(0.01)  # 模拟时间间隔
        track.update(bbox, confidence=1.0)

    speed_x, speed_y = analyzer.calculate_speed(track, use_real_world=True)
    direction = analyzer.calculate_direction(track)
    stats = analyzer.analyze_track(track)

    print(f"速度: ({speed_x:.3f}, {speed_y:.3f}) m/s")
    print(f"方向: {direction:.2f}°")
    print("轨迹统计信息:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def demo_model_selection() -> None:
    print("\n=== 自动模型选择示例（ModelSelector） ===")
    result = select_model(
        model_type="detection",
        accuracy=75,
        speed=60,
        memory=256,
        task="general_detection",
    )
    if "model_name" in result:
        print("推荐模型:", result["model_name"])
        info = result["model_info"]
        print(
            f"  类型: {info['type'].value}, 准确率: {info['accuracy']}, "
            f"速度: {info['speed']}, 内存: {info['memory']} MB"
        )
    else:
        print("无法根据当前硬件信息找到合适模型:", result.get("error"))


def demo_multimodal_fusion() -> None:
    print("\n=== 多模态融合示例（MultimodalFusion） ===")
    np.random.seed(0)
    # 构造视觉和文本特征
    vision_feat = np.random.randn(4, 16).astype("float32")
    text_feat = np.random.randn(4, 32).astype("float32")

    fused = fuse_features(
        [vision_feat, text_feat],
        fusion_type="concat",
        hidden_dim=32,
        output_dim=10,
        device="cpu",
        dropout=0.0,
    )
    print("融合后特征形状:", fused.shape)


if __name__ == "__main__":
    demo_model_optimization()
    demo_fine_tuning_config()
    demo_data_augmentation()
    demo_trajectory_analysis()
    demo_model_selection()
    demo_multimodal_fusion()

