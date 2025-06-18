import os
import tarfile
import zipfile

import requests


def download_file(url, save_path):
    """Download file from URL."""
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)


def extract_tar(file_path, target_dir):
    """Extract tar file."""
    with tarfile.open(file_path, "r") as tar:
        tar.extractall()


def extract_zip(file_path, target_dir):
    """Extract zip file."""
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)


def download_tiny_imagenet(save_dir):
    """Download Tiny ImageNet dataset."""
    os.makedirs(save_dir, exist_ok=True)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    save_path = os.path.join(save_dir, "tiny-imagenet-200.zip")
    download_file(url, save_path)

    # Extract the downloaded tar file
    extract_zip(save_path, save_dir)


if __name__ == "__main__":
    download_tiny_imagenet("./eval/data/")
