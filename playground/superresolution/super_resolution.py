import os
import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from cv2 import dnn_superres
from loguru import logger

class SuperResolutionModel:
    """Base class for all super-resolution models"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scale_factor = 4

    def load_model(self):
        raise NotImplementedError("Subclasses must implement load_model")

    def upscale(self, img):
        """
        Upscale an image using the model

        Args:
            img (numpy.ndarray): Input image in BGR format (OpenCV default)

        Returns:
            numpy.ndarray: Upscaled image
        """
        raise NotImplementedError("Subclasses must implement upscale")


class FSRCNN(SuperResolutionModel):
    """FSRCNN model from OpenCV's DNN module"""

    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor

        # Create model directory if it doesn't exist
        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)

        # Set model paths based on scale factor
        if scale_factor == 2:
            self.model_path = model_dir / 'FSRCNN_x2.pb'
            self.model_url = "https://github.com/fannymonori/TF-FSRCNN/raw/master/FSRCNN/FSRCNN_x2.pb"
        elif scale_factor == 3:
            self.model_path = model_dir / 'FSRCNN_x3.pb'
            self.model_url = "https://github.com/fannymonori/TF-FSRCNN/raw/master/FSRCNN/FSRCNN_x3.pb"
        elif scale_factor == 4:
            self.model_path = model_dir / 'FSRCNN_x4.pb'
            self.model_url = "https://github.com/fannymonori/TF-FSRCNN/raw/master/FSRCNN/FSRCNN_x4.pb"
        else:
            raise ValueError(f"Unsupported scale factor: {scale_factor}")

        self.load_model()

    def load_model(self):
        """Load or download the model"""
        if not self.model_path.exists():
            print(f"Downloading FSRCNN x{self.scale_factor} model...")
            self._download_model()

        # Load the model
        self.model = cv2.dnn_superres.DnnSuperResImpl_create()
        self.model.readModel(str(self.model_path))
        self.model.setModel("fsrcnn", self.scale_factor)

        print(f"FSRCNN x{self.scale_factor} model loaded successfully")

    def _download_model(self):
        """Download the model from GitHub"""
        import urllib.request

        print(f"Downloading model from {self.model_url}")
        urllib.request.urlretrieve(self.model_url, self.model_path)
        print(f"Model downloaded to {self.model_path}")

    def upscale(self, img):
        """
        Upscale the image using FSRCNN

        Args:
            img (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Upscaled image
        """
        return self.model.upsample(img)


class EDSR(SuperResolutionModel):
    """EDSR model from OpenCV's DNN module"""

    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor

        # Create model directory if it doesn't exist
        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)

        # Set model paths based on scale factor
        if scale_factor == 2:
            self.model_path = model_dir / 'EDSR_x2.pb'
            self.model_url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb"
        elif scale_factor == 3:
            self.model_path = model_dir / 'EDSR_x3.pb'
            self.model_url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x3.pb"
        elif scale_factor == 4:
            self.model_path = model_dir / 'EDSR_x4.pb'
            self.model_url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
        else:
            raise ValueError(f"Unsupported scale factor: {scale_factor}")

        self.load_model()

    def load_model(self):
        """Load or download the model"""
        if not self.model_path.exists():
            print(f"Downloading EDSR x{self.scale_factor} model...")
            self._download_model()

        # Load the model
        self.model = cv2.dnn_superres.DnnSuperResImpl_create()
        self.model.readModel(str(self.model_path))
        self.model.setModel("edsr", self.scale_factor)

        print(f"EDSR x{self.scale_factor} model loaded successfully")

    def _download_model(self):
        """Download the model from GitHub"""
        import urllib.request

        print(f"Downloading model from {self.model_url}")
        urllib.request.urlretrieve(self.model_url, self.model_path)
        print(f"Model downloaded to {self.model_path}")

    def upscale(self, img):
        """
        Upscale the image using EDSR

        Args:
            img (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Upscaled image
        """
        return self.model.upsample(img)


def enhance_aerial_image(img, sr_model, sharpen_factor=0.5, denoise=True, enhance_contrast=True):
    """
    Apply additional enhancements specific to aerial imagery

    Args:
        img (numpy.ndarray): Input image
        sr_model: Super-resolution model
        sharpen_factor (float): Sharpening factor (0.0 to 1.0)
        denoise (bool): Apply denoising to reduce noise
        enhance_contrast (bool): Apply contrast enhancement

    Returns:
        numpy.ndarray: Enhanced image
    """
    # Apply denoising if requested
    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)

    # Apply super-resolution
    upscaled = sr_model.upscale(img)

    # Apply sharpening if requested
    if sharpen_factor > 0:
        blurred = cv2.GaussianBlur(upscaled, (0, 0), 3)
        upscaled = cv2.addWeighted(upscaled, 1 + sharpen_factor, blurred, -sharpen_factor, 0)

    # Apply contrast enhancement if requested
    if enhance_contrast:
        # Convert to LAB color space
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)

        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to BGR
        upscaled = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return upscaled


def super_resolve(input_path, output_path, model_type='edsr', scale_factor=4,
                  sharpen=0.5, denoise=True, enhance_contrast=True):
    """
    Main function for super-resolution

    Args:
        input_path (Path or str): Path to input image
        output_path (Path or str): Path to output image
        model_type (str): Type of super-resolution model ('fsrcnn', 'edsr')
        scale_factor (int): Upscaling factor (2, 3, or 4)
        sharpen (float): Sharpening factor (0.0 to 1.0)
        denoise (bool): Apply denoising
        enhance_contrast (bool): Apply contrast enhancement

    Returns:
        bool: Success flag
    """
    # Convert string paths to Path objects
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Ensure input file exists
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return False

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the super-resolution model
    try:
        # Import needed for DNN module
        cv2.dnn_superres = __import__('cv2.dnn_superres', fromlist=[''])

        if model_type.lower() == 'fsrcnn':
            sr_model = FSRCNN(scale_factor)
        elif model_type.lower() == 'edsr':
            sr_model = EDSR(scale_factor)
        else:
            print(f"Error: Unsupported model type: {model_type}")
            print("Supported types: 'fsrcnn', 'edsr'")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to traditional upscaling...")
        sr_model = None

    # Read the input image
    try:
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"Error: Could not read image {input_path}")
            return False
    except Exception as e:
        print(f"Error reading image: {e}")
        return False

    # Process the image
    try:
        if sr_model is not None:
            # Use DNN-based super-resolution
            result = enhance_aerial_image(img, sr_model, sharpen, denoise, enhance_contrast)
        else:
            # Fallback to OpenCV's traditional upscaling
            h, w = img.shape[:2]
            result = cv2.resize(img, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)

            # Apply sharpening and contrast enhancement
            if sharpen > 0:
                blurred = cv2.GaussianBlur(result, (0, 0), 3)
                result = cv2.addWeighted(result, 1 + sharpen, blurred, -sharpen, 0)

            if enhance_contrast:
                lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return False

    # Save the result
    try:
        cv2.imwrite(str(output_path), result)
        return True
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return False


if __name__ == "__main__":
    # Example usage:
    super_resolve(
        input_path=Path(
            "/Volumes/G-DRIVE/DD_MS_COG_ALL_TILES/herdnet_112/super_resolve/Scruz_SCM01-02-03_04012023_centered_799719_9917912.jpg"),
        output_path=Path(
            "/Volumes/G-DRIVE/DD_MS_COG_ALL_TILES/herdnet_112/super_resolve/Scruz_SCM01-02-03_04012023_centered_799719_9917912_224.jpg"),
        scale_factor=2, model_type='edsr',
    )