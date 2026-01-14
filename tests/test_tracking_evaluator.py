"""
Unit tests for TrackingEvaluator

Tests standard MOT metrics calculation:
- MOTA (Multiple Object Tracking Accuracy)
- MOTP (Multiple Object Tracking Precision)
- IDF1 (ID F1 Score)
"""

import sys
from pathlib import Path
import importlib.util

# Setup minimal package structure
repo_root = Path(__file__).resolve().parents[1]
pkg_path = repo_root / "visionframework"


def test_tracking_evaluator_basic():
    """Test basic TrackingEvaluator instantiation and MOTA calculation"""
    import types

    # Create minimal module structure to avoid heavy imports
    sys.modules.setdefault("visionframework", types.SimpleNamespace(__path__=[str(pkg_path)]))
    sys.modules.setdefault("visionframework.utils", types.SimpleNamespace(__path__=[str(pkg_path / "utils")]))
    sys.modules.setdefault("visionframework.utils.evaluation", types.SimpleNamespace(__path__=[str(pkg_path / "utils" / "evaluation")]))
    sys.modules.setdefault("visionframework.data", types.SimpleNamespace(__path__=[str(pkg_path / "data")]))

    # Load tracking_evaluator by file path to avoid package-level imports
    evaluator_path = pkg_path / "utils" / "evaluation" / "tracking_evaluator.py"
    spec = importlib.util.spec_from_file_location("tracking_evaluator_module", str(evaluator_path))
    mod = importlib.util.module_from_spec(spec)
    
    # Provide minimal Track class
    class MinimalTrack:
        pass
    
    # Mock Track in sys.modules
    track_module = types.SimpleNamespace(Track=MinimalTrack)
    sys.modules["visionframework.data.track"] = track_module
    
    spec.loader.exec_module(mod)
    TrackingEvaluator = mod.TrackingEvaluator
    
    # Test instantiation
    evaluator = TrackingEvaluator(iou_threshold=0.5)
    assert evaluator is not None
    assert evaluator.iou_threshold == 0.5


def test_tracking_evaluator_mota():
    """Test MOTA calculation with simple test case"""
    import types

    sys.modules.setdefault("visionframework", types.SimpleNamespace(__path__=[str(pkg_path)]))
    sys.modules.setdefault("visionframework.utils", types.SimpleNamespace(__path__=[str(pkg_path / "utils")]))
    sys.modules.setdefault("visionframework.utils.evaluation", types.SimpleNamespace(__path__=[str(pkg_path / "utils" / "evaluation")]))
    sys.modules.setdefault("visionframework.data", types.SimpleNamespace(__path__=[str(pkg_path / "data")]))

    evaluator_path = pkg_path / "utils" / "evaluation" / "tracking_evaluator.py"
    spec = importlib.util.spec_from_file_location("tracking_evaluator_module", str(evaluator_path))
    mod = importlib.util.module_from_spec(spec)
    
    class MinimalTrack:
        pass
    
    track_module = types.SimpleNamespace(Track=MinimalTrack)
    sys.modules["visionframework.data.track"] = track_module
    
    spec.loader.exec_module(mod)
    TrackingEvaluator = mod.TrackingEvaluator
    
    evaluator = TrackingEvaluator(iou_threshold=0.5)
    
    # Test case: 2 frames, each with 1 prediction and 1 ground truth
    # Frame 0: pred_track_id=1, gt_track_id=1 (match)
    # Frame 1: pred_track_id=1, gt_track_id=1 (match)
    pred_tracks = [
        [
            {
                "track_id": 1,
                "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
            }
        ],
        [
            {
                "track_id": 1,
                "bbox": {"x1": 11, "y1": 11, "x2": 51, "y2": 51}
            }
        ]
    ]
    
    gt_tracks = [
        [
            {
                "track_id": 1,
                "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
            }
        ],
        [
            {
                "track_id": 1,
                "bbox": {"x1": 11, "y1": 11, "x2": 51, "y2": 51}
            }
        ]
    ]
    
    result = evaluator.calculate_mota(pred_tracks, gt_tracks)
    
    assert "MOTA" in result
    assert "total_gt" in result
    assert result["total_gt"] == 2
    assert result["total_fp"] == 0
    assert result["total_fn"] == 0
    # Perfect match should give MOTA close to 1.0
    assert result["MOTA"] >= 0.9


def test_tracking_evaluator_idf1():
    """Test IDF1 calculation"""
    import types

    sys.modules.setdefault("visionframework", types.SimpleNamespace(__path__=[str(pkg_path)]))
    sys.modules.setdefault("visionframework.utils", types.SimpleNamespace(__path__=[str(pkg_path / "utils")]))
    sys.modules.setdefault("visionframework.utils.evaluation", types.SimpleNamespace(__path__=[str(pkg_path / "utils" / "evaluation")]))
    sys.modules.setdefault("visionframework.data", types.SimpleNamespace(__path__=[str(pkg_path / "data")]))

    evaluator_path = pkg_path / "utils" / "evaluation" / "tracking_evaluator.py"
    spec = importlib.util.spec_from_file_location("tracking_evaluator_module", str(evaluator_path))
    mod = importlib.util.module_from_spec(spec)
    
    class MinimalTrack:
        pass
    
    track_module = types.SimpleNamespace(Track=MinimalTrack)
    sys.modules["visionframework.data.track"] = track_module
    
    spec.loader.exec_module(mod)
    TrackingEvaluator = mod.TrackingEvaluator
    
    evaluator = TrackingEvaluator()
    
    # Simple test: 1 frame with 1 detection matching
    pred_tracks = [
        [
            {
                "track_id": 1,
                "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
            }
        ]
    ]
    
    gt_tracks = [
        [
            {
                "track_id": 1,
                "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
            }
        ]
    ]
    
    result = evaluator.calculate_idf1(pred_tracks, gt_tracks)
    
    assert "IDF1" in result
    assert "IDTP" in result
    assert "IDFP" in result
    assert "IDFN" in result
    # Perfect match should give high IDF1
    assert result["IDF1"] >= 0.9


def test_tracking_evaluator_motp():
    """Test MOTP calculation"""
    import types

    sys.modules.setdefault("visionframework", types.SimpleNamespace(__path__=[str(pkg_path)]))
    sys.modules.setdefault("visionframework.utils", types.SimpleNamespace(__path__=[str(pkg_path / "utils")]))
    sys.modules.setdefault("visionframework.utils.evaluation", types.SimpleNamespace(__path__=[str(pkg_path / "utils" / "evaluation")]))
    sys.modules.setdefault("visionframework.data", types.SimpleNamespace(__path__=[str(pkg_path / "data")]))

    evaluator_path = pkg_path / "utils" / "evaluation" / "tracking_evaluator.py"
    spec = importlib.util.spec_from_file_location("tracking_evaluator_module", str(evaluator_path))
    mod = importlib.util.module_from_spec(spec)
    
    class MinimalTrack:
        pass
    
    track_module = types.SimpleNamespace(Track=MinimalTrack)
    sys.modules["visionframework.data.track"] = track_module
    
    spec.loader.exec_module(mod)
    TrackingEvaluator = mod.TrackingEvaluator
    
    evaluator = TrackingEvaluator()
    
    # Test: identical boxes should give distance = 0
    pred_tracks = [
        [
            {
                "track_id": 1,
                "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
            }
        ]
    ]
    
    gt_tracks = [
        [
            {
                "track_id": 1,
                "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
            }
        ]
    ]
    
    result = evaluator.calculate_motp(pred_tracks, gt_tracks)
    
    assert "MOTP" in result
    assert "total_matched_pairs" in result
    assert result["total_matched_pairs"] == 1
    # Identical boxes should give distance close to 0
    assert result["MOTP"] < 1.0


def test_tracking_evaluator_comprehensive():
    """Test comprehensive evaluate() method"""
    import types

    sys.modules.setdefault("visionframework", types.SimpleNamespace(__path__=[str(pkg_path)]))
    sys.modules.setdefault("visionframework.utils", types.SimpleNamespace(__path__=[str(pkg_path / "utils")]))
    sys.modules.setdefault("visionframework.utils.evaluation", types.SimpleNamespace(__path__=[str(pkg_path / "utils" / "evaluation")]))
    sys.modules.setdefault("visionframework.data", types.SimpleNamespace(__path__=[str(pkg_path / "data")]))

    evaluator_path = pkg_path / "utils" / "evaluation" / "tracking_evaluator.py"
    spec = importlib.util.spec_from_file_location("tracking_evaluator_module", str(evaluator_path))
    mod = importlib.util.module_from_spec(spec)
    
    class MinimalTrack:
        pass
    
    track_module = types.SimpleNamespace(Track=MinimalTrack)
    sys.modules["visionframework.data.track"] = track_module
    
    spec.loader.exec_module(mod)
    TrackingEvaluator = mod.TrackingEvaluator
    
    evaluator = TrackingEvaluator()
    
    pred_tracks = [
        [
            {
                "track_id": 1,
                "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
            }
        ]
    ]
    
    gt_tracks = [
        [
            {
                "track_id": 1,
                "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
            }
        ]
    ]
    
    result = evaluator.evaluate(pred_tracks, gt_tracks)
    
    assert "MOTA" in result
    assert "MOTP" in result
    assert "IDF1" in result
    assert "precision" in result
    assert "recall" in result
    assert "details" in result


def main():
    """Run all tests"""
    tests = [
        ("TrackingEvaluator instantiation", test_tracking_evaluator_basic),
        ("MOTA calculation", test_tracking_evaluator_mota),
        ("IDF1 calculation", test_tracking_evaluator_idf1),
        ("MOTP calculation", test_tracking_evaluator_motp),
        ("Comprehensive evaluation", test_tracking_evaluator_comprehensive),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
