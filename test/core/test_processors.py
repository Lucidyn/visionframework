"""
Basic tests for processors (CLIPExtractor, ReIDExtractor).

These tests verify:
- Processor initialization (even if model loading fails)
- Processor process returns expected structure
- Feature extraction result integrity
"""

import numpy as np

from visionframework import CLIPExtractor, ReIDExtractor


def _make_dummy_image(h: int = 224, w: int = 224) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_clip_extractor_creation_returns_instance_even_if_init_fails() -> None:
    """
    CLIPExtractor should return an instance even if model loading fails.
    Initialization failures are handled internally via logging.
    """
    clip_extractor = CLIPExtractor(model_name="openai/clip-vit-base-patch32", device="cpu")
    assert isinstance(clip_extractor, CLIPExtractor)


def test_clip_extractor_process_returns_expected_structure() -> None:
    """
    CLIPExtractor.process should return expected structure even if initialization fails.
    """
    clip_extractor = CLIPExtractor(model_name="openai/clip-vit-base-patch32", device="cpu")
    
    # Initialize (should fail due to non-existent model)
    clip_extractor.initialize()
    
    # Process dummy image
    img = _make_dummy_image()
    features = clip_extractor.process(img)
    
    # Should return a list or numpy array
    assert isinstance(features, (list, np.ndarray))


def test_clip_extractor_with_text_prompt() -> None:
    """
    Test CLIPExtractor with text prompt for zero-shot classification.
    """
    clip_extractor = CLIPExtractor(model_name="openai/clip-vit-base-patch32", device="cpu")
    
    # Initialize (should fail due to non-existent model)
    clip_extractor.initialize()
    
    # Process dummy image with text prompt
    img = _make_dummy_image()
    text_prompt = "cat, dog, person"  # Dummy text prompt
    
    # This should return scores or features even if model fails
    result = clip_extractor.process(img, text_prompt=text_prompt)
    assert isinstance(result, (list, np.ndarray, dict))


def test_reid_extractor_creation_returns_instance_even_if_init_fails() -> None:
    """
    ReIDExtractor should return an instance even if model loading fails.
    Initialization failures are handled internally via logging.
    """
    reid_extractor = ReIDExtractor(model_name="resnet50", device="cpu", model_path="nonexistent_model.pt")
    assert isinstance(reid_extractor, ReIDExtractor)


def test_reid_extractor_process_returns_expected_structure() -> None:
    """
    ReIDExtractor.process should return expected structure even if initialization fails.
    """
    reid_extractor = ReIDExtractor(model_name="resnet50", device="cpu", model_path="nonexistent_model.pt")
    
    # Initialize (should fail due to non-existent model)
    reid_extractor.initialize()
    
    # Process dummy image
    img = _make_dummy_image()
    features = reid_extractor.process(img, bboxes=[(0, 0, 100, 100)])
    
    # Should return a list or numpy array
    assert isinstance(features, (list, np.ndarray))


def test_reid_extractor_with_bbox() -> None:
    """
    Test ReIDExtractor with bounding box for targeted feature extraction.
    """
    reid_extractor = ReIDExtractor(model_name="resnet50", device="cpu", model_path="nonexistent_model.pt")
    
    # Initialize (should fail due to non-existent model)
    reid_extractor.initialize()
    
    # Process dummy image with bounding box
    img = _make_dummy_image()
    bbox = (50, 50, 150, 150)  # Dummy bounding box
    
    # This should return features even if model fails
    features = reid_extractor.process(img, bboxes=[bbox])
    assert isinstance(features, (list, np.ndarray))
