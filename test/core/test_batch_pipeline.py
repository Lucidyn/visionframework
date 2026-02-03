"""
Basic tests for BatchPipeline.

These tests verify:
- BatchPipeline initialization (even if model loading fails)
- BatchPipeline process returns expected structure
- Batch processing functionality with multiple images
"""

import numpy as np

from visionframework import BatchPipeline


def _make_dummy_image(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_batch_pipeline_creation_returns_instance_even_if_init_fails() -> None:
    """
    BatchPipeline should return an instance even if model loading fails.
    Initialization failures are handled internally via logging.
    """
    config = {
        "detector_config": {
            "model_path": "nonexistent_model.pt",  # Intentionally non-existent path
            "device": "cpu",
            "conf_threshold": 0.5,
        },
        "enable_tracking": False,
        "batch_size": 4,
    }
    batch_pipeline = BatchPipeline(config)
    assert isinstance(batch_pipeline, BatchPipeline)


def test_batch_pipeline_process_returns_expected_structure() -> None:
    """
    BatchPipeline.process should return expected structure even if initialization fails.
    """
    config = {
        "detector_config": {
            "model_path": "nonexistent_model.pt",  # Intentionally non-existent path
            "device": "cpu",
            "conf_threshold": 0.5,
        },
        "enable_tracking": False,
        "batch_size": 4,
    }
    batch_pipeline = BatchPipeline(config)
    
    # Initialize (should fail due to non-existent model)
    batch_pipeline.initialize()
    
    # Create a batch of dummy images
    batch_size = 2
    batch_images = [_make_dummy_image() for _ in range(batch_size)]
    
    # Process batch
    results = batch_pipeline.process_batch(batch_images)
    
    # Should return a list of dicts with expected keys
    assert isinstance(results, list)
    assert len(results) == batch_size
    
    for result in results:
        assert isinstance(result, dict)
        assert "detections" in result
        assert "tracks" in result
        assert "poses" in result
        assert isinstance(result["detections"], list)
        assert isinstance(result["tracks"], list)
        assert isinstance(result["poses"], list)


def test_batch_pipeline_with_different_batch_sizes() -> None:
    """
    Test BatchPipeline with different batch sizes.
    """
    config = {
        "detector_config": {
            "model_path": "nonexistent_model.pt",  # Intentionally non-existent path
            "device": "cpu",
            "conf_threshold": 0.5,
        },
        "enable_tracking": False,
        "batch_size": 8,  # Larger batch size
    }
    batch_pipeline = BatchPipeline(config)
    
    # Initialize (should fail due to non-existent model)
    batch_pipeline.initialize()
    
    # Test with batch size smaller than configured
    small_batch_size = 3
    small_batch_images = [_make_dummy_image() for _ in range(small_batch_size)]
    small_batch_results = batch_pipeline.process_batch(small_batch_images)
    assert isinstance(small_batch_results, list)
    assert len(small_batch_results) == small_batch_size
    
    # Test with batch size larger than configured
    large_batch_size = 10
    large_batch_images = [_make_dummy_image() for _ in range(large_batch_size)]
    large_batch_results = batch_pipeline.process_batch(large_batch_images)
    assert isinstance(large_batch_results, list)
    assert len(large_batch_results) == large_batch_size


def test_batch_pipeline_with_empty_batch() -> None:
    """
    Test BatchPipeline with empty batch.
    """
    config = {
        "detector_config": {
            "model_path": "nonexistent_model.pt",  # Intentionally non-existent path
            "device": "cpu",
            "conf_threshold": 0.5,
        },
        "enable_tracking": False,
        "batch_size": 4,
    }
    batch_pipeline = BatchPipeline(config)
    
    # Initialize (should fail due to non-existent model)
    batch_pipeline.initialize()
    
    # Process empty batch
    empty_batch_results = batch_pipeline.process_batch([])
    assert isinstance(empty_batch_results, list)
    assert len(empty_batch_results) == 0
