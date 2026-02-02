"""
Image augmentation functionality

This module provides functionality for augmenting images for model training.
"""

import random
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import cv2


class AugmentationType(Enum):
    """Augmentation type enum"""
    FLIP = "flip"
    ROTATE = "rotate"
    SCALE = "scale"
    TRANSLATE = "translate"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    HUE = "hue"
    BLUR = "blur"
    NOISE = "noise"
    CUTOUT = "cutout"
    MIXUP = "mixup"
    CUTMIX = "cutmix"
    COLOR_JITTER = "color_jitter"
    RANDOM_ERASE = "random_erase"


class InterpolationType(Enum):
    """Interpolation type enum"""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


@dataclass
class AugmentationConfig:
    """
    Augmentation configuration class
    
    Attributes:
        augmentations: List of augmentations to apply
        probability: Probability of applying each augmentation
        random_order: Whether to apply augmentations in random order
        interpolation: Interpolation method to use
        seed: Random seed for reproducibility
    """
    augmentations: List[AugmentationType]
    probability: float = 0.5
    random_order: bool = True
    interpolation: InterpolationType = InterpolationType.BILINEAR
    seed: Optional[int] = None


class ImageAugmenter:
    """
    Image augmenter class for applying various augmentations to images
    """
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize image augmenter
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
        
        # Set random seed if provided
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
        
        # Map interpolation type to OpenCV interpolation
        self.interpolation_map = {
            InterpolationType.NEAREST: cv2.INTER_NEAREST,
            InterpolationType.BILINEAR: cv2.INTER_LINEAR,
            InterpolationType.BICUBIC: cv2.INTER_CUBIC,
            InterpolationType.LANCZOS: cv2.INTER_LANCZOS4
        }
    
    def augment(
        self,
        image: Union[Image.Image, np.ndarray],
        **kwargs
    ) -> Union[Image.Image, np.ndarray]:
        """
        Augment image
        
        Args:
            image: Image to augment
            **kwargs: Additional augmentation parameters
        
        Returns:
            Augmented image
        """
        # Convert PIL Image to numpy array if needed
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            image = np.array(image)
        
        # Apply augmentations
        augmentations = self.config.augmentations.copy()
        
        # Shuffle augmentations if random order is enabled
        if self.config.random_order:
            random.shuffle(augmentations)
        
        # Apply each augmentation
        for augmentation in augmentations:
            if random.random() < self.config.probability:
                image = self._apply_augmentation(image, augmentation, **kwargs)
        
        # Convert back to PIL Image if needed
        if is_pil:
            image = Image.fromarray(image)
        
        return image
    
    def _apply_augmentation(
        self,
        image: np.ndarray,
        augmentation: AugmentationType,
        **kwargs
    ) -> np.ndarray:
        """
        Apply specific augmentation to image
        
        Args:
            image: Image to augment
            augmentation: Augmentation to apply
            **kwargs: Additional augmentation parameters
        
        Returns:
            Augmented image
        """
        if augmentation == AugmentationType.FLIP:
            return self._flip(image, **kwargs)
        elif augmentation == AugmentationType.ROTATE:
            return self._rotate(image, **kwargs)
        elif augmentation == AugmentationType.SCALE:
            return self._scale(image, **kwargs)
        elif augmentation == AugmentationType.TRANSLATE:
            return self._translate(image, **kwargs)
        elif augmentation == AugmentationType.BRIGHTNESS:
            return self._adjust_brightness(image, **kwargs)
        elif augmentation == AugmentationType.CONTRAST:
            return self._adjust_contrast(image, **kwargs)
        elif augmentation == AugmentationType.SATURATION:
            return self._adjust_saturation(image, **kwargs)
        elif augmentation == AugmentationType.HUE:
            return self._adjust_hue(image, **kwargs)
        elif augmentation == AugmentationType.BLUR:
            return self._blur(image, **kwargs)
        elif augmentation == AugmentationType.NOISE:
            return self._add_noise(image, **kwargs)
        elif augmentation == AugmentationType.CUTOUT:
            return self._cutout(image, **kwargs)
        elif augmentation == AugmentationType.MIXUP:
            return self._mixup(image, **kwargs)
        elif augmentation == AugmentationType.CUTMIX:
            return self._cutmix(image, **kwargs)
        elif augmentation == AugmentationType.COLOR_JITTER:
            return self._color_jitter(image, **kwargs)
        elif augmentation == AugmentationType.RANDOM_ERASE:
            return self._random_erase(image, **kwargs)
        else:
            return image
    
    def _flip(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Flip image horizontally or vertically
        
        Args:
            image: Image to flip
            **kwargs: Additional flip parameters
        
        Returns:
            Flipped image
        """
        flip_type = kwargs.get("flip_type", random.choice(["horizontal", "vertical"]))
        
        if flip_type == "horizontal":
            return cv2.flip(image, 1)
        elif flip_type == "vertical":
            return cv2.flip(image, 0)
        elif flip_type == "both":
            return cv2.flip(image, -1)
        else:
            return image
    
    def _rotate(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Rotate image
        
        Args:
            image: Image to rotate
            **kwargs: Additional rotate parameters
        
        Returns:
            Rotated image
        """
        angle = kwargs.get("angle", random.uniform(-30, 30))
        border_mode = kwargs.get("border_mode", cv2.BORDER_CONSTANT)
        border_value = kwargs.get("border_value", 0)
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust rotation matrix to account for translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation
        return cv2.warpAffine(
            image,
            rotation_matrix,
            (new_width, new_height),
            flags=self.interpolation_map[self.config.interpolation],
            borderMode=border_mode,
            borderValue=border_value
        )
    
    def _scale(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Scale image
        
        Args:
            image: Image to scale
            **kwargs: Additional scale parameters
        
        Returns:
            Scaled image
        """
        scale_factor = kwargs.get("scale_factor", random.uniform(0.8, 1.2))
        
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=self.interpolation_map[self.config.interpolation]
        )
    
    def _translate(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Translate image
        
        Args:
            image: Image to translate
            **kwargs: Additional translate parameters
        
        Returns:
            Translated image
        """
        translate_x = kwargs.get("translate_x", random.uniform(-0.1, 0.1))
        translate_y = kwargs.get("translate_y", random.uniform(-0.1, 0.1))
        border_mode = kwargs.get("border_mode", cv2.BORDER_CONSTANT)
        border_value = kwargs.get("border_value", 0)
        
        height, width = image.shape[:2]
        tx = int(translate_x * width)
        ty = int(translate_y * height)
        
        # Create translation matrix
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # Apply translation
        return cv2.warpAffine(
            image,
            translation_matrix,
            (width, height),
            borderMode=border_mode,
            borderValue=border_value
        )
    
    def _adjust_brightness(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Adjust image brightness
        
        Args:
            image: Image to adjust
            **kwargs: Additional brightness parameters
        
        Returns:
            Brightness-adjusted image
        """
        brightness_factor = kwargs.get("brightness_factor", random.uniform(0.5, 1.5))
        
        # Convert to PIL Image for brightness adjustment
        image_pil = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(image_pil)
        image_pil = enhancer.enhance(brightness_factor)
        
        return np.array(image_pil)
    
    def _adjust_contrast(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Adjust image contrast
        
        Args:
            image: Image to adjust
            **kwargs: Additional contrast parameters
        
        Returns:
            Contrast-adjusted image
        """
        contrast_factor = kwargs.get("contrast_factor", random.uniform(0.5, 1.5))
        
        # Convert to PIL Image for contrast adjustment
        image_pil = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(image_pil)
        image_pil = enhancer.enhance(contrast_factor)
        
        return np.array(image_pil)
    
    def _adjust_saturation(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Adjust image saturation
        
        Args:
            image: Image to adjust
            **kwargs: Additional saturation parameters
        
        Returns:
            Saturation-adjusted image
        """
        saturation_factor = kwargs.get("saturation_factor", random.uniform(0.5, 1.5))
        
        # Convert to PIL Image for saturation adjustment
        image_pil = Image.fromarray(image)
        enhancer = ImageEnhance.Color(image_pil)
        image_pil = enhancer.enhance(saturation_factor)
        
        return np.array(image_pil)
    
    def _adjust_hue(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Adjust image hue
        
        Args:
            image: Image to adjust
            **kwargs: Additional hue parameters
        
        Returns:
            Hue-adjusted image
        """
        hue_factor = kwargs.get("hue_factor", random.uniform(-0.1, 0.1))
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Adjust hue
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_factor * 180) % 180
        
        # Convert back to RGB
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def _blur(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Blur image
        
        Args:
            image: Image to blur
            **kwargs: Additional blur parameters
        
        Returns:
            Blurred image
        """
        blur_kernel = kwargs.get("blur_kernel", random.choice([3, 5, 7]))
        
        return cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    
    def _add_noise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Add noise to image
        
        Args:
            image: Image to add noise to
            **kwargs: Additional noise parameters
        
        Returns:
            Noisy image
        """
        noise_level = kwargs.get("noise_level", random.uniform(0, 0.1))
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level * 255, image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    def _cutout(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply cutout augmentation to image
        
        Args:
            image: Image to augment
            **kwargs: Additional cutout parameters
        
        Returns:
            Cutout-augmented image
        """
        cutout_size = kwargs.get("cutout_size", random.uniform(0.1, 0.3))
        num_cutouts = kwargs.get("num_cutouts", random.randint(1, 3))
        fill_value = kwargs.get("fill_value", 0)
        
        height, width = image.shape[:2]
        cutout_height = int(height * cutout_size)
        cutout_width = int(width * cutout_size)
        
        augmented_image = image.copy()
        
        for _ in range(num_cutouts):
            # Random position
            x = random.randint(0, width - cutout_width)
            y = random.randint(0, height - cutout_height)
            
            # Apply cutout
            augmented_image[y:y+cutout_height, x:x+cutout_width] = fill_value
        
        return augmented_image
    
    def _mixup(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply mixup augmentation to image
        
        Args:
            image: Image to augment
            **kwargs: Additional mixup parameters
        
        Returns:
            Mixup-augmented image
        """
        # Mixup requires a second image
        other_image = kwargs.get("other_image", None)
        if other_image is None:
            return image
        
        alpha = kwargs.get("alpha", random.uniform(0.1, 0.9))
        
        # Ensure both images have the same shape
        if image.shape != other_image.shape:
            other_image = cv2.resize(
                other_image,
                (image.shape[1], image.shape[0]),
                interpolation=self.interpolation_map[self.config.interpolation]
            )
        
        # Apply mixup
        mixed_image = alpha * image + (1 - alpha) * other_image
        mixed_image = mixed_image.astype(np.uint8)
        
        return mixed_image
    
    def _cutmix(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply cutmix augmentation to image
        
        Args:
            image: Image to augment
            **kwargs: Additional cutmix parameters
        
        Returns:
            Cutmix-augmented image
        """
        # Cutmix requires a second image
        other_image = kwargs.get("other_image", None)
        if other_image is None:
            return image
        
        cutmix_size = kwargs.get("cutmix_size", random.uniform(0.2, 0.5))
        
        height, width = image.shape[:2]
        cutmix_height = int(height * cutmix_size)
        cutmix_width = int(width * cutmix_size)
        
        # Ensure both images have the same shape
        if image.shape != other_image.shape:
            other_image = cv2.resize(
                other_image,
                (width, height),
                interpolation=self.interpolation_map[self.config.interpolation]
            )
        
        # Random position
        x = random.randint(0, width - cutmix_width)
        y = random.randint(0, height - cutmix_height)
        
        # Apply cutmix
        augmented_image = image.copy()
        augmented_image[y:y+cutmix_height, x:x+cutmix_width] = other_image[y:y+cutmix_height, x:x+cutmix_width]
        
        return augmented_image
    
    def _color_jitter(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply color jitter augmentation to image
        
        Args:
            image: Image to augment
            **kwargs: Additional color jitter parameters
        
        Returns:
            Color-jittered image
        """
        # Color jitter is a combination of brightness, contrast, saturation, and hue adjustments
        brightness = kwargs.get("brightness", random.uniform(0.8, 1.2))
        contrast = kwargs.get("contrast", random.uniform(0.8, 1.2))
        saturation = kwargs.get("saturation", random.uniform(0.8, 1.2))
        hue = kwargs.get("hue", random.uniform(-0.1, 0.1))
        
        # Convert to PIL Image for color jitter
        image_pil = Image.fromarray(image)
        
        # Adjust brightness
        enhancer = ImageEnhance.Brightness(image_pil)
        image_pil = enhancer.enhance(brightness)
        
        # Adjust contrast
        enhancer = ImageEnhance.Contrast(image_pil)
        image_pil = enhancer.enhance(contrast)
        
        # Adjust saturation
        enhancer = ImageEnhance.Color(image_pil)
        image_pil = enhancer.enhance(saturation)
        
        # Adjust hue
        if hue != 0:
            image_pil = ImageOps.colorize(
                ImageOps.grayscale(image_pil),
                black="black",
                white="white"
            )
        
        return np.array(image_pil)
    
    def _random_erase(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply random erase augmentation to image
        
        Args:
            image: Image to augment
            **kwargs: Additional random erase parameters
        
        Returns:
            Random-erase-augmented image
        """
        erase_size = kwargs.get("erase_size", random.uniform(0.02, 0.3))
        aspect_ratio = kwargs.get("aspect_ratio", random.uniform(0.3, 3.0))
        fill_value = kwargs.get("fill_value", 0)
        
        height, width = image.shape[:2]
        area = height * width
        erase_area = area * erase_size
        
        # Calculate erase region dimensions
        erase_height = int(np.sqrt(erase_area / aspect_ratio))
        erase_width = int(np.sqrt(erase_area * aspect_ratio))
        
        # Ensure dimensions are valid
        erase_height = min(erase_height, height)
        erase_width = min(erase_width, width)
        
        # Random position
        x = random.randint(0, width - erase_width)
        y = random.randint(0, height - erase_height)
        
        # Apply random erase
        augmented_image = image.copy()
        augmented_image[y:y+erase_height, x:x+erase_width] = fill_value
        
        return augmented_image


def augment_image(
    image: Union[Image.Image, np.ndarray],
    augmentations: List[Union[str, AugmentationType]],
    **kwargs
) -> Union[Image.Image, np.ndarray]:
    """
    Augment image
    
    Args:
        image: Image to augment
        augmentations: List of augmentations to apply
        **kwargs: Additional augmentation parameters
    
    Returns:
        Augmented image
    """
    # Convert augmentation strings to AugmentationType enums
    augmentation_types = []
    for aug in augmentations:
        if isinstance(aug, str):
            augmentation_types.append(AugmentationType(aug))
        else:
            augmentation_types.append(aug)
    
    # Create augmentation config
    config = AugmentationConfig(
        augmentations=augmentation_types,
        **kwargs
    )
    
    # Create augmenter
    augmenter = ImageAugmenter(config)
    
    # Augment image
    return augmenter.augment(image, **kwargs)


def get_default_augmentations() -> List[AugmentationType]:
    """
    Get default augmentations
    
    Returns:
        List of default augmentations
    """
    return [
        AugmentationType.FLIP,
        AugmentationType.ROTATE,
        AugmentationType.SCALE,
        AugmentationType.BRIGHTNESS,
        AugmentationType.CONTRAST,
        AugmentationType.SATURATION,
        AugmentationType.BLUR,
        AugmentationType.CUTOUT
    ]


def get_heavy_augmentations() -> List[AugmentationType]:
    """
    Get heavy augmentations
    
    Returns:
        List of heavy augmentations
    """
    return [
        AugmentationType.FLIP,
        AugmentationType.ROTATE,
        AugmentationType.SCALE,
        AugmentationType.TRANSLATE,
        AugmentationType.BRIGHTNESS,
        AugmentationType.CONTRAST,
        AugmentationType.SATURATION,
        AugmentationType.HUE,
        AugmentationType.BLUR,
        AugmentationType.NOISE,
        AugmentationType.CUTOUT,
        AugmentationType.COLOR_JITTER,
        AugmentationType.RANDOM_ERASE
    ]


def get_light_augmentations() -> List[AugmentationType]:
    """
    Get light augmentations
    
    Returns:
        List of light augmentations
    """
    return [
        AugmentationType.FLIP,
        AugmentationType.BRIGHTNESS,
        AugmentationType.CONTRAST,
        AugmentationType.SATURATION
    ]
