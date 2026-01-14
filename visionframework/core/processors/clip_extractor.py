"""
CLIP Feature Extractor

Provides CLIP-based image/text embedding and zero-shot classification.
"""

from typing import List, Optional, Any, Union
import numpy as np
from .feature_extractor import FeatureExtractor


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
    
    def initialize(self) -> None:
        """Initialize CLIP model and processor."""
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
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
            self.model.to(device)
