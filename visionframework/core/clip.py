"""
CLIP integration module

Provides a simple wrapper around HuggingFace's CLIP model (or compatible)
to extract image/text embeddings and perform simple zero-shot classification
and similarity search.

This module keeps dependencies optional: if `transformers` or `torch` are
not available the module will fail on `initialize()` with a clear message.
"""

from typing import List, Optional, Any
import numpy as np

# Try relative import first (normal package import), then fallback for direct file imports
try:
    from ..utils.io.config import ModelCache, DeviceManager
    from ..utils.monitoring.logger import get_logger
except ImportError:
    # Fallback for direct file imports or edge cases
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from visionframework.utils.io.config import ModelCache, DeviceManager
    from visionframework.utils.monitoring.logger import get_logger

logger = get_logger(__name__)


class CLIPExtractor:
    """Wrapper around a CLIP model (HuggingFace transformers API)

    Features:
    - `initialize()` loads model and processor
    - `encode_image()` returns normalized image embedding (numpy)
    - `encode_text()` returns normalized text embeddings (numpy)
    - `image_text_similarity()` returns cosine similarities
    - `zero_shot_classify()` scores candidate labels for an image
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.model_name = self.config.get("model_name", "openai/clip-vit-base-patch32")
        self.device = self.config.get("device", "cpu")
        self.use_fp16 = bool(self.config.get("use_fp16", False))

        self.model = None
        self.processor = None
        self.is_initialized = False

    def initialize(self) -> bool:
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            proc_key = f"clip_proc:{self.model_name}"
            mdl_key = f"clip_model:{self.model_name}"

            self.processor = ModelCache.get_model(proc_key, lambda: CLIPProcessor.from_pretrained(self.model_name))
            self.model = ModelCache.get_model(mdl_key, lambda: CLIPModel.from_pretrained(self.model_name))

            device = DeviceManager.normalize_device(self.device)
            if device != self.device:
                logger.info(f"Requested device '{self.device}' not available, using '{device}' instead")
            self.device = device
            try:
                if hasattr(self.model, 'to'):
                    self.model.to(self.device)
            except Exception:
                logger.debug("Model.to(device) not supported or failed; continuing")
            try:
                if hasattr(self.model, 'eval'):
                    self.model.eval()
            except Exception:
                pass
            self.is_initialized = True
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CLIP model: {e}")

    def cleanup(self) -> None:
        try:
            # Release cached resources if present
            # ModelCache keys used when initialized
            proc_key = f"clip_proc:{self.model_name}"
            mdl_key = f"clip_model:{self.model_name}"
            try:
                ModelCache.release_model(mdl_key)
            except Exception:
                pass
            try:
                ModelCache.release_model(proc_key)
            except Exception:
                pass
            self.model = None
            self.processor = None
        finally:
            self.is_initialized = False

    def encode_image(self, image: Any) -> np.ndarray:
        """Encode a single image (numpy array BGR or RGB) or a list of images.

        Returns:
            np.ndarray: shape (N, D) where D is embedding dim
        """
        if not self.is_initialized:
            self.initialize()

        import torch

        # Accept list or single
        is_batch = isinstance(image, (list, tuple))
        imgs = image if is_batch else [image]

        # CLIPProcessor expects PIL or numpy in RGB
        inputs = self.processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # inference with optional autocast on cuda
        try:
            if self.use_fp16 and self.device == 'cuda':
                ctx = torch.cuda.amp.autocast()
            else:
                ctx = torch.no_grad()
        except Exception:
            ctx = torch.no_grad()

        with ctx:
            outputs = self.model.get_image_features(**inputs)

        feats = outputs.cpu().numpy()
        # normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats = feats / norms
        return feats if is_batch else feats[0:1]

    def encode_text(self, texts: List[str]) -> np.ndarray:
        if not self.is_initialized:
            self.initialize()

        import torch

        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        feats = outputs.cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats = feats / norms
        return feats

    def image_text_similarity(self, image: Any, texts: List[str]) -> np.ndarray:
        img_emb = self.encode_image(image)
        txt_emb = self.encode_text(texts)

        # img_emb: (N, D) - when single image we return (1,D); txt_emb: (M, D)
        # return cosine similarity matrix shape (N, M)
        sim = np.matmul(img_emb, txt_emb.T)
        return sim

    def zero_shot_classify(self, image: Any, candidate_labels: List[str]) -> List[float]:
        sim = self.image_text_similarity(image, candidate_labels)
        # if single image, sim shape is (1, M)
        scores = sim[0].tolist() if sim.shape[0] == 1 else sim.mean(axis=0).tolist()
        return scores
