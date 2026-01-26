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
    - Supports multiple CLIP model variants
    - Embedding caching for improved performance
    - Batch processing optimization
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.model_name = self.config.get("model_name", "openai/clip-vit-base-patch32")
        self.device = self.config.get("device", "cpu")
        self.use_fp16 = bool(self.config.get("use_fp16", False))
        self.cache_enabled = bool(self.config.get("cache_enabled", True))
        self.max_cache_size = int(self.config.get("max_cache_size", 1000))
        self.preprocess_options = self.config.get("preprocess_options", {})

        self.model = None
        self.processor = None
        self.is_initialized = False
        
        # Embedding cache
        self._image_cache = {}
        self._text_cache = {}
        
        # Supported CLIP model architectures
        self.supported_models = [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-large-patch14",
            "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        ]

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

        Args:
            image: Single image (numpy array) or list of images

        Returns:
            np.ndarray: shape (N, D) where D is embedding dim
        """
        if not self.is_initialized:
            self.initialize()

        import torch
        from PIL import Image as PILImage
        import cv2

        # Accept list or single
        is_batch = isinstance(image, (list, tuple))
        imgs = image if is_batch else [image]
        
        # Check cache first if enabled
        if self.cache_enabled:
            # Create cache keys for images
            cache_keys = []
            for img in imgs:
                if isinstance(img, PILImage.Image):
                    # Convert PIL image to numpy for hashing
                    img_np = np.array(img)
                    cache_key = hash(img_np.tostring())
                elif isinstance(img, np.ndarray):
                    cache_key = hash(img.tostring())
                else:
                    cache_key = None
                cache_keys.append(cache_key)
            
            # Check if all images are in cache
            all_cached = all(key is not None and key in self._image_cache for key in cache_keys)
            if all_cached:
                feats = np.stack([self._image_cache[key] for key in cache_keys])
                return feats if is_batch else feats[0:1]
        
        # Image preprocessing
        processed_imgs = []
        for img in imgs:
            processed_img = img.copy() if isinstance(img, np.ndarray) else img
            
            # Apply preprocessing options if specified
            if self.preprocess_options:
                if isinstance(processed_img, np.ndarray):
                    # Convert BGR to RGB if needed (assuming OpenCV BGR format)
                    if self.preprocess_options.get("bgr_to_rgb", True):
                        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    
                    # Resize if specified
                    if "resize" in self.preprocess_options:
                        size = self.preprocess_options["resize"]
                        processed_img = cv2.resize(processed_img, size)
                    
                    # Convert back to PIL for CLIP processor
                    processed_img = PILImage.fromarray(processed_img)
            
            processed_imgs.append(processed_img)
        
        # CLIPProcessor expects PIL or numpy in RGB
        inputs = self.processor(images=processed_imgs, return_tensors="pt")
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
        
        # Update cache if enabled
        if self.cache_enabled:
            for i, (key, feat) in enumerate(zip(cache_keys, feats)):
                if key is not None:
                    # Check cache size and evict if necessary
                    if len(self._image_cache) >= self.max_cache_size:
                        # Remove oldest item (FIFO)
                        oldest_key = next(iter(self._image_cache))
                        del self._image_cache[oldest_key]
                    self._image_cache[key] = feat
        
        return feats if is_batch else feats[0:1]

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode list of text strings into embeddings.

        Args:
            texts: List of text strings to encode

        Returns:
            np.ndarray: shape (N, D) where D is embedding dim
        """
        if not self.is_initialized:
            self.initialize()

        import torch

        # Check cache first if enabled
        if self.cache_enabled:
            # Create cache keys for texts
            cache_keys = [hash(text) for text in texts]
            
            # Check if all texts are in cache
            all_cached = all(key in self._text_cache for key in cache_keys)
            if all_cached:
                return np.stack([self._text_cache[key] for key in cache_keys])
        
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Use autocast if available and enabled
        try:
            if self.use_fp16 and self.device == 'cuda':
                ctx = torch.cuda.amp.autocast()
            else:
                ctx = torch.no_grad()
        except Exception:
            ctx = torch.no_grad()
        
        with ctx:
            outputs = self.model.get_text_features(**inputs)

        feats = outputs.cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats = feats / norms
        
        # Update cache if enabled
        if self.cache_enabled:
            for text, key, feat in zip(texts, cache_keys, feats):
                # Check cache size and evict if necessary
                if len(self._text_cache) >= self.max_cache_size:
                    # Remove oldest item (FIFO)
                    oldest_key = next(iter(self._text_cache))
                    del self._text_cache[oldest_key]
                self._text_cache[key] = feat
        
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
    
    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """Clear embedding cache.
        
        Args:
            cache_type: Type of cache to clear, 'image', 'text', or None for all
        """
        if cache_type is None or cache_type == 'image':
            self._image_cache.clear()
        if cache_type is None or cache_type == 'text':
            self._text_cache.clear()
    
    def get_cache_status(self) -> Dict[str, int]:
        """Get cache status.
        
        Returns:
            Dict with cache sizes: {'image': int, 'text': int}
        """
        return {
            'image': len(self._image_cache),
            'text': len(self._text_cache)
        }
    
    def filter_detections_by_text(self, image: np.ndarray, detections: List[Any], 
                                 text_description: str, threshold: float = 0.5) -> List[Any]:
        """Filter detections based on text description similarity.
        
        Args:
            image: Original image
            detections: List of detection objects with bbox coordinates
            text_description: Text description to match
            threshold: Similarity threshold for filtering
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        import cv2
        
        # Extract image patches for each detection
        image_patches = []
        for det in detections:
            # Assume detection has bbox attribute (x1, y1, x2, y2)
            bbox = getattr(det, 'bbox', None)
            if bbox is None:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            # Extract patch
            patch = image[y1:y2, x1:x2]
            if patch.size > 0:
                image_patches.append(patch)
            else:
                image_patches.append(image)  # Fallback to full image if patch is invalid
        
        if not image_patches:
            return []
        
        # Encode patches and text
        patch_embeddings = self.encode_image(image_patches)
        text_embedding = self.encode_text([text_description])[0]
        
        # Calculate similarities
        similarities = np.dot(patch_embeddings, text_embedding)
        
        # Filter detections based on similarity threshold
        filtered_detections = []
        for det, sim in zip(detections, similarities):
            if sim >= threshold:
                # Add similarity score to detection
                setattr(det, 'clip_similarity', float(sim))
                filtered_detections.append(det)
        
        return filtered_detections
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported CLIP models.
        
        Returns:
            List of supported model names
        """
        return self.supported_models.copy()
