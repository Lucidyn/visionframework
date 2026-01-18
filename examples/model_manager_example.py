"""
model_manager_example.py
Show usage of ModelManager to get or download model files.

Usage:
  python examples/model_manager_example.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from visionframework import get_model_manager


def main():
    manager = get_model_manager()
    path = manager.get_model_path("yolov8n.pt", download=False)
    print(f"Model path (may be None if not cached): {path}")


if __name__ == "__main__":
    main()
