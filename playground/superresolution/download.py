import os
import requests
import torch
from tqdm import tqdm
from pathlib import Path

# URL for the pre-trained ESRGAN model weights - using the official model from ESRGAN authors
# This is a 4x upscaling model specifically trained for photo-realistic enhancement
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('content-length', 0))

    if response.status_code != 200:
        print(f"Failed to download model: Status code {response.status_code}")
        return False

    # Ensure directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # Download with progress bar
    progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {os.path.basename(destination)}")

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    progress_bar.close()
    return True


def download_model():
    """Download the pre-trained model"""
    # Create models directory in the current working directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "RealESRGAN_x4plus.pth"

    if model_path.exists():
        print(f"Model already exists at {model_path}")
        return model_path

    print("Downloading pre-trained ESRGAN model...")
    success = download_file(MODEL_URL, model_path)

    if success:
        print(f"Successfully downloaded model to {model_path}")
        return model_path
    else:
        print("Failed to download model")
        return None


if __name__ == "__main__":
    download_model()