"""
CLIP Feature Extractor

Provides CLIP-based image/text embedding and zero-shot classification.
"""

from typing import List, Optional, Any, Union
import numpy as np
from .feature_extractor import FeatureExtractor
from ...utils.config import ModelCache, DeviceManager
from ...utils.logger import get_logger

logger = get_logger(__name__)


class CLIPExtractor(FeatureExtractor):
    """
    CLIP feature extractor for image and text embeddings.
    
    Features:
    - Image embedding extraction
    - Text embedding extraction  
    - Image-text similarity computation
    - Zero-shot classification
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", 
                 device: str = "cpu", use_fp16: bool = False):
        """
        Initialize CLIP extractor.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ("cpu", "cuda", etc.)
            use_fp16: Whether to use FP16 precision
        """
        super().__init__(model_name, device)
        self.use_fp16 = use_fp16
        self.model = None
        self.processor = None
        self._cached_model_key: Optional[str] = None
        self._cached_processor_key: Optional[str] = None
    
    def initialize(self) -> None:
        """Initialize CLIP model and processor."""
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            # Use cache for model and processor
            proc_key = f"clip_proc:{self.model_name}"
            mdl_key = f"clip_model:{self.model_name}"

            self.processor = ModelCache.get_model(proc_key, lambda: CLIPProcessor.from_pretrained(self.model_name))
            self._cached_processor_key = proc_key
            self.model = ModelCache.get_model(mdl_key, lambda: CLIPModel.from_pretrained(self.model_name))
            self._cached_model_key = mdl_key

            # Normalize device and move model if supported
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
            self._initialized = True
        except ImportError as e:
            raise ImportError(f"CLIP dependencies not installed: {e}. "
                            "Install with: pip install 'visionframework[clip]'")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CLIP model: {e}")
    
    def extract(self, input_data: Any) -> Union[np.ndarray, dict]:
        """
        Extract features from input data.
        
        Args:
            input_data: Image array or text string/list
        
        Returns:
            Feature embedding(s) as numpy array
        """
        if isinstance(input_data, str):
            return self.encode_text([input_data])
        elif isinstance(input_data, list) and isinstance(input_data[0], str):
            return self.encode_text(input_data)
        else:
            return self.encode_image(input_data)
    
    def encode_image(self, image: Any) -> np.ndarray:
        """
        Encode image(s) to embeddings.
        
        Args:
            image: Single image or list of images (numpy arrays)
        
        Returns:
            Normalized embeddings shape (N, D)
        """
        if not self.is_initialized():
            self.initialize()
        
        import torch
        
        # Handle batch or single image
        is_batch = isinstance(image, (list, tuple))
        imgs = image if is_batch else [image]
        
        # Process images
        inputs = self.processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference with optional FP16
        with torch.no_grad():
            if self.use_fp16 and self.device == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.model.get_image_features(**inputs)
            else:
                outputs = self.model.get_image_features(**inputs)
        
        # Normalize embeddings
        feats = outputs.cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats = feats / norms
        
        return feats if is_batch else feats[0:1]
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode text(s) to embeddings.
        
        Args:
            texts: List of text strings
        
        Returns:
            Normalized embeddings shape (N, D)
        """
        if not self.is_initialized():
            self.initialize()
        
        import torch
        
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if self.use_fp16 and self.device == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.model.get_text_features(**inputs)
            else:
                outputs = self.model.get_text_features(**inputs)
        
        # Normalize embeddings
        feats = outputs.cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats = feats / norms
        
        return feats
    
    def image_text_similarity(self, image: Any, texts: List[str]) -> np.ndarray:
        """
        Compute similarity between image and texts.
        
        Args:
            image: Image array
            texts: List of text strings
        
        Returns:
            Similarity matrix shape (num_images, num_texts)
        """
        img_emb = self.encode_image(image)
        txt_emb = self.encode_text(texts)
        
        # Cosine similarity
        sim = np.matmul(img_emb, txt_emb.T)
        return sim
    
    def zero_shot_classify(self, image: Any, candidate_labels: List[str]) -> List[float]:
        """
        Perform zero-shot classification.
        
        Args:
            image: Image to classify
            candidate_labels: List of candidate class labels
        
        Returns:
            Similarity scores for each label
        """
        sim = self.image_text_similarity(image, candidate_labels)
        scores = sim[0].tolist() if sim.shape[0] == 1 else sim.mean(axis=0).tolist()
        return scores
    
    def _move_to_device(self, device: str) -> None:
        """Move model to device."""
        if self.model is not None:
            try:
                self.model.to(device)
            except Exception:
                logger.debug("CLIP model move to device failed")

    def cleanup(self) -> None:
        """Release or release-reference cached CLIP model and processor."""
        try:
            if self._cached_model_key:
                try:
                    ModelCache.release_model(self._cached_model_key)
                except Exception as e:
                    logger.warning(f"Error releasing cached CLIP model '{self._cached_model_key}': {e}")
                self._cached_model_key = None
                self.model = None
            else:
                if self.model is not None:
                    try:
                        if hasattr(self.model, 'to'):
                            self.model.to('cpu')
                    except Exception:
                        pass
                    try:
                        del self.model
                    except Exception:
                        self.model = None
                    self.model = None

            if self._cached_processor_key:
                try:
                    ModelCache.release_model(self._cached_processor_key)
                except Exception as e:
                    logger.warning(f"Error releasing cached CLIP processor '{self._cached_processor_key}': {e}")
                self._cached_processor_key = None
                self.processor = None
            else:
                if self.processor is not None:
                    try:
                        del self.processor
                    except Exception:
                        self.processor = None
                    self.processor = None

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        finally:
            self._initialized = False
