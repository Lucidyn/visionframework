"""
15 - 批量图像处理
=================
使用 Vision.process_batch() 一次处理多张图像。

适用场景：
  - 离线数据集推理
  - 批量标注生成
  - 性能基准测试
"""

import time
import cv2
import numpy as np
from visionframework import Vision, ResultExporter


def load_images(paths: list) -> list:
    """加载图像列表，找不到的用随机图像替代。"""
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  未找到 {p}，使用随机图像替代")
            img = np.random.randint(0, 200, (480, 640, 3), dtype=np.uint8)
        images.append(img)
    return images


def main() -> None:
    # ── 准备图像 ──
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
    images = load_images(image_paths)
    print(f"已加载 {len(images)} 张图像")

    # ── 创建 Vision 实例 ──
    v = Vision(model="yolov8n.pt", conf=0.25)
    print("Vision 配置:", v.info())

    # ── 批量处理 ──
    t0 = time.perf_counter()
    results = v.process_batch(images)
    elapsed = time.perf_counter() - t0

    print(f"\n批量推理完成，耗时 {elapsed:.3f}s，"
          f"平均 {elapsed / len(images) * 1000:.1f}ms/张")

    # ── 汇总结果 ──
    total_dets = 0
    for i, result in enumerate(results):
        dets = result.get("detections", [])
        total_dets += len(dets)
        print(f"  图像 {i+1}: {len(dets)} 个检测")
        for det in dets[:3]:  # 只打印前 3 个
            print(f"    {det.class_name}: {det.confidence:.2f}  bbox={det.bbox}")

    print(f"\n共检测到 {total_dets} 个目标")

    # ── 导出结果 ──
    exporter = ResultExporter()
    for i, (img, result) in enumerate(zip(images, results)):
        dets = result.get("detections", [])
        if dets:
            exporter.export_detections(dets, f"batch_result_{i}.json")
            print(f"  已导出 batch_result_{i}.json")

    # ── 可视化第一张 ──
    if results:
        annotated = v.draw(images[0], results[0])
        cv2.imshow("Batch Result [0]", annotated)
        print("\n按任意键关闭...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    v.cleanup()


if __name__ == "__main__":
    main()
