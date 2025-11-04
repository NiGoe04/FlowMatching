from datetime import datetime

import torch
from torch import Tensor
from torchvision import datasets, transforms
import os

from src.flow_matching.model.velocity_model_basic import SimpleVelocityModel
from src.flow_matching.model.velocity_model_unet import UnetVelocityModel


def store_model(save_dir_path, approach_name, model):
    """
    Saves the given model in the specified directory and returns the full model path.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    os.makedirs(save_dir_path, exist_ok=True)

    # Construct filename with timestamp and extension
    filename = f"model_{approach_name}_{timestamp}.pth"

    # Full path for saving
    full_path = os.path.join(save_dir_path, filename)

    # Save model state dict
    torch.save(model.state_dict(), full_path)

    return full_path

def load_model_n_dim(dim, model_path, device="cpu"):
    # Create an instance of the model architecture
    model = SimpleVelocityModel(dim=dim)

    # Load the saved weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Set model to evaluation mode by default
    model.to(device)
    model.eval()

    return model

def load_model_unet(model_path, dropout_rate, device="cpu"):
    """
    Loads a UNet-based velocity field model from disk.

    Args:
        model_path (str): Path to the saved model weights.
        dropout_rate (float): Dropout rate used when constructing the UNet model.
        device (str): 'cpu' or 'cuda'.

    Returns:
        model (nn.Module): The loaded UNet model in eval mode.
    """
    # Instantiate the model architecture
    model = UnetVelocityModel(dropout_rate=dropout_rate)

    # Load the saved weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move to device and set eval mode
    model.to(device)
    model.eval()

    return model


def load_mnist_tensor(size_set: int = 1000, train: bool = True, device='cpu') -> Tensor:
    """
    Loads a subset of the MNIST dataset and returns it as a tensor of shape [N, 1, H, W].

    Args:
        size_set (int): Number of samples to return.
        train (bool): Whether to load the training or test set.
        device (str): 'cpu' or 'cuda' device for the tensor.

    Returns:
        Tensor: MNIST images as tensor [N, 1, H, W], dtype=torch.float32, normalized to [0,1].
    """
    # Prepare path
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../datasets/mnist')
    os.makedirs(dataset_dir, exist_ok=True)  # create folder if not exists

    # Transform to tensor and normalize to [0,1]
    transform = transforms.ToTensor()

    # Check if dataset already exists
    download_flag = not os.path.exists(os.path.join(dataset_dir, 'MNIST/raw'))

    mnist_dataset = datasets.MNIST(root=dataset_dir, train=train, download=download_flag, transform=transform)

    # Take only first `size_set` samples
    size_set = min(size_set, len(mnist_dataset))
    images = [mnist_dataset[i][0] for i in range(size_set)]

    # Stack into one tensor of shape [N, 1, H, W]
    images_tensor = torch.stack(images).to(device)

    return images_tensor