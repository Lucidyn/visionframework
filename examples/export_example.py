"""
export_example.py
Demonstrate exporting detection results to JSON/CSV using framework utilities.

Usage:
  python examples/export_example.py
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parents[1]))
from visionframework import ResultExporter
from visionframework.data.detection import Detection


def main():
    # Create dummy detections
    detections = [
        Detection((10, 10, 50, 50), 0.9, 0, "person"),
        Detection((100, 100, 150, 150), 0.8, 1, "car"),
    ]

    exporter = ResultExporter({"output_dir": "example_outputs", "formats": ["json", "csv"]})
    exporter.initialize()
    exporter.export_detections(detections, "run1")
    print("Exported detections to example_outputs/")


if __name__ == "__main__":
    main()
