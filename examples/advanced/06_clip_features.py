"""
06 - CLIP 特征提取
==================
使用 CLIP 模型进行：
- 图像特征提取
- 文本特征提取
- 图文相似度计算
- 零样本分类

依赖：pip install transformers
"""

import cv2
import numpy as np
from visionframework import CLIPExtractor


def main() -> None:
    # ── 1. 创建 CLIP 提取器 ──
    clip = CLIPExtractor(
        model_name="openai/clip-vit-base-patch32",
        device="cpu",  # 或 "cuda"
    )
    clip.initialize()
    print(f"CLIP 初始化完成, device={clip.device}")

    # ── 2. 准备图像 ──
    # 用测试图像或随机图像
    img = cv2.imread("test.jpg")
    if img is None:
        print("未找到 test.jpg，使用随机图像演示")
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # ── 3. 图像编码 ──
    img_emb = clip.encode_image(img)
    print(f"\n图像嵌入维度: {img_emb.shape}")  # (1, 512)

    # ── 4. 文本编码 ──
    texts = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
    txt_emb = clip.encode_text(texts)
    print(f"文本嵌入维度: {txt_emb.shape}")  # (3, 512)

    # ── 5. 图文相似度 ──
    sim = clip.image_text_similarity(img, texts)
    print(f"\n图文相似度:")
    for text, score in zip(texts, sim[0]):
        print(f"  '{text}': {score:.4f}")

    # ── 6. 零样本分类 ──
    labels = ["cat", "dog", "car", "person", "tree", "building"]
    scores = clip.zero_shot_classify(img, labels)
    print(f"\n零样本分类结果:")
    ranked = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
    for label, score in ranked:
        print(f"  {label}: {score:.4f}")

    # ── 7. 批量图像编码 ──
    batch = [img, np.fliplr(img).copy(), np.flipud(img).copy()]
    batch_emb = clip.encode_image(batch)
    print(f"\n批量编码: {len(batch)} 张图像 -> {batch_emb.shape}")

    # ── 8. 通用 extract 接口 ──
    # 传入图像 -> 返回图像嵌入
    feat_img = clip.extract(img)
    print(f"\nextract(image) 维度: {feat_img.shape}")

    # 传入文本 -> 返回文本嵌入
    feat_txt = clip.extract(["hello world"])
    print(f"extract(text) 维度: {feat_txt.shape}")

    # ── 清理 ──
    clip.cleanup()
    print("\nCLIP 资源已释放")


if __name__ == "__main__":
    main()
