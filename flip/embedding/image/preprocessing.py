"""Image preprocessing utilities."""

from typing import Tuple, Optional
import numpy as np

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ImagePreprocessor:
    """Preprocessing pipeline for images before embedding."""
    
    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        enhance_contrast: bool = False,
        denoise: bool = False
    ):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target size (width, height) or None to keep original
            normalize: Whether to normalize pixel values
            enhance_contrast: Whether to enhance contrast
            denoise: Whether to apply denoising
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow is required for image preprocessing")
        
        self.target_size = target_size
        self.normalize = normalize
        self.enhance_contrast = enhance_contrast
        self.denoise = denoise
    
    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to RGB if needed
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Resize if target size specified
        if self.target_size:
            image = self._resize(image, self.target_size)
        
        # Denoise
        if self.denoise:
            image = self._denoise(image)
        
        # Enhance contrast
        if self.enhance_contrast:
            image = self._enhance_contrast(image)
        
        return image
    
    def _resize(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Resize image maintaining aspect ratio.
        
        Args:
            image: PIL Image
            target_size: Target (width, height)
            
        Returns:
            Resized image
        """
        # Calculate aspect ratio
        width, height = image.size
        target_width, target_height = target_size
        
        # Maintain aspect ratio
        aspect = width / height
        target_aspect = target_width / target_height
        
        if aspect > target_aspect:
            # Width is limiting factor
            new_width = target_width
            new_height = int(target_width / aspect)
        else:
            # Height is limiting factor
            new_height = target_height
            new_width = int(target_height * aspect)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _denoise(self, image: Image.Image) -> Image.Image:
        """
        Apply denoising filter.
        
        Args:
            image: PIL Image
            
        Returns:
            Denoised image
        """
        return image.filter(ImageFilter.MedianFilter(size=3))
    
    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """
        Enhance image contrast.
        
        Args:
            image: PIL Image
            
        Returns:
            Contrast-enhanced image
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.2)  # 20% contrast increase
    
    def batch_preprocess(self, images: list) -> list:
        """
        Preprocess multiple images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of preprocessed images
        """
        return [self.preprocess(img) for img in images]


class ImageAugmentor:
    """Image augmentation for training/testing."""
    
    @staticmethod
    def rotate(image: Image.Image, angle: float) -> Image.Image:
        """Rotate image by angle."""
        return image.rotate(angle, expand=True)
    
    @staticmethod
    def flip_horizontal(image: Image.Image) -> Image.Image:
        """Flip image horizontally."""
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    
    @staticmethod
    def flip_vertical(image: Image.Image) -> Image.Image:
        """Flip image vertically."""
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    
    @staticmethod
    def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
        """
        Adjust image brightness.
        
        Args:
            image: PIL Image
            factor: Brightness factor (1.0 = original, <1.0 = darker, >1.0 = brighter)
            
        Returns:
            Brightness-adjusted image
        """
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def add_noise(image: Image.Image, noise_level: float = 0.1) -> Image.Image:
        """
        Add random noise to image.
        
        Args:
            image: PIL Image
            noise_level: Noise level (0.0 to 1.0)
            
        Returns:
            Noisy image
        """
        img_array = np.array(image).astype(np.float32)
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)
