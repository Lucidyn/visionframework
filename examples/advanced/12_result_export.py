"""
12 - 结果导出
=============
使用 Vision 检测后，将结果导出为 JSON / CSV。
"""

import os
from visionframework import Vision, ResultExporter, Detection, Track


def export_dummy_results(exporter: ResultExporter, output_dir: str) -> None:
    """使用虚拟数据演示导出格式。"""
    detections = [
        Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="person"),
        Detection(bbox=(300, 150, 400, 250), confidence=0.85, class_id=1, class_name="car"),
    ]
    tracks = [
        Track(track_id=1, bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="person"),
        Track(track_id=2, bbox=(300, 150, 400, 250), confidence=0.85, class_id=1, class_name="car"),
    ]

    exporter.export_detections_to_json(detections, os.path.join(output_dir, "detections.json"))
    exporter.export_detections_to_csv(detections, os.path.join(output_dir, "detections.csv"))
    exporter.export_tracks_to_json(tracks, os.path.join(output_dir, "tracks.json"))
    exporter.export_tracks_to_csv(tracks, os.path.join(output_dir, "tracks.csv"))
    print(f"虚拟结果已导出到 {output_dir}/")


def export_real_detections(exporter: ResultExporter, output_dir: str) -> None:
    """使用真实检测结果导出。"""
    v = Vision(model="yolov8n.pt", conf=0.25)

    source = "test.jpg"  # 图片 / 文件夹 / 视频
    all_detections: list = []

    for frame, meta, result in v.run(source):
        dets = result["detections"]
        all_detections.extend(dets)
        print(f"[{meta.get('source_path')}] {len(dets)} 个检测")

    if all_detections:
        os.makedirs(output_dir, exist_ok=True)
        exporter.export_detections_to_json(all_detections, os.path.join(output_dir, "real_detections.json"))
        exporter.export_detections_to_csv(all_detections, os.path.join(output_dir, "real_detections.csv"))
        print(f"结果已导出到 {output_dir}/")
    else:
        print("未检测到任何目标。")


def main() -> None:
    output_dir = "exports"
    os.makedirs(output_dir, exist_ok=True)
    exporter = ResultExporter()

    print("=== 1. 虚拟数据导出 ===")
    export_dummy_results(exporter, output_dir)

    print("\n=== 2. 真实检测结果导出 ===")
    export_real_detections(exporter, os.path.join(output_dir, "real"))


if __name__ == "__main__":
    main()
