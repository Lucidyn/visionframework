"""
16 - 模型优化：量化与剪枝
==========================
演示如何使用量化和剪枝工具压缩 PyTorch 模型。

工具：
  - QuantizationConfig / quantize_model  — 动态/静态量化
  - PruningConfig / prune_model          — L1/L2/随机剪枝
"""

import torch
import torch.nn as nn

from visionframework import QuantizationConfig, quantize_model, PruningConfig, prune_model


# ── 示例模型 ──
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_nonzero(model: nn.Module) -> int:
    return sum(p.count_nonzero().item() for p in model.parameters())


def main() -> None:
    model = SimpleClassifier()
    x = torch.randn(4, 512)

    print(f"原始模型参数量: {count_params(model):,}")
    out_orig = model(x)
    print(f"原始输出形状: {out_orig.shape}")

    # ── 1. 动态量化 ──
    print("\n── 动态量化 ──")
    q_cfg = QuantizationConfig(quantization_type="dynamic", verbose=True)
    q_model = quantize_model(model, q_cfg)
    out_q = q_model(x)
    print(f"量化后输出形状: {out_q.shape}")
    print(f"最大差异: {(out_orig - out_q).abs().max().item():.6f}")

    # ── 2. L1 非结构化剪枝 ──
    print("\n── L1 非结构化剪枝 (amount=0.3) ──")
    p_cfg = PruningConfig(pruning_type="l1_unstructured", amount=0.3, verbose=True)
    p_model = prune_model(model, p_cfg)
    nonzero_before = count_nonzero(model)
    nonzero_after  = count_nonzero(p_model)
    print(f"非零参数: {nonzero_before:,} → {nonzero_after:,}")
    out_p = p_model(x)
    print(f"剪枝后输出形状: {out_p.shape}")

    # ── 3. 全局剪枝 ──
    print("\n── 全局剪枝 (amount=0.5) ──")
    gp_cfg = PruningConfig(amount=0.5, global_pruning=True, verbose=True)
    gp_model = prune_model(SimpleClassifier(), gp_cfg)
    print(f"全局剪枝后非零参数: {count_nonzero(gp_model):,}")

    print("\n模型优化演示完成！")


if __name__ == "__main__":
    main()
