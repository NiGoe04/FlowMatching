import math
import os
from datetime import datetime
from typing import Tuple

import torch
from torch import Tensor
from torchvision import datasets, transforms

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
    model = SimpleVelocityModel(dim=dim, device=device)

    # Load the saved weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Set model to evaluation mode by default
    model.to(device)
    model.eval()

    return model

def load_model_unet(model_path, dropout_rate, device: torch.device):
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
    model = UnetVelocityModel(dropout_rate=dropout_rate, device=device)

    # Load the saved weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move to device and set eval mode
    model.to(device)
    model.eval()

    return model


def load_mnist_tensor(size_set: int, device: torch.device = "cpu", set_type: str = "MNIST_N", train: bool = True) -> Tensor:
    """
    Loads a subset of either the standard MNIST or Fashion-MNIST dataset and returns it as a tensor of shape [N, 1, H, W].

    Args:
        size_set (int): Number of samples to return.
        device (torch.device): Device ('cpu' or 'cuda') to load the data onto.
        set_type (str): "MNIST_N" for standard MNIST, "MNIST_F" for Fashion-MNIST.
        train (bool): Whether to load the training or test set.

    Returns:
        Tensor: Images as tensor [N, 1, H, W], dtype=torch.float32, normalized to [0,1].
    """
    # Prepare dataset directory
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../datasets')
    os.makedirs(dataset_dir, exist_ok=True)

    # Transform to tensor and normalize to [0,1]
    transform = transforms.ToTensor()

    # Determine which dataset to load
    if set_type == "MNIST_N":
        dataset_name = "mnist"
        dataset_class = datasets.MNIST
    elif set_type == "MNIST_F":
        dataset_name = "fashion_mnist"
        dataset_class = datasets.FashionMNIST
    else:
        raise ValueError(f"Invalid dataset type '{set_type}'. Use 'MNIST_N' or 'MNIST_F'.")

    # Full path for this dataset
    dataset_path = os.path.join(dataset_dir, dataset_name)

    # Only download if the dataset directory doesn't exist yet
    download_flag = not os.path.exists(os.path.join(dataset_path, 'raw'))

    # Load the dataset
    dataset = dataset_class(root=dataset_path, train=train, download=download_flag, transform=transform)

    # Take only first `size_set` samples
    size_set = min(size_set, len(dataset))
    images = [dataset[i][0] for i in range(size_set)]

    # Stack into one tensor of shape [N, 1, H, W]
    images_tensor = torch.stack(images).to(device)

    return images_tensor

def get_velocity_field_tensor_2d(time_range: Tuple, num_times, bounds, density, model, device) -> Tensor:
    """
    :param time_range: time range (from, to)
    :param num_times: amount times
    :param bounds: bounding box size (bottom_left, top_right)
    :param density: value density
    :param model: the velocity model to sample from
    :param device: the device ('cpu' or 'cuda')
    :return: [T, H, W, D] tensor
    """
    from_time, to_time = time_range
    time_grid = torch.linspace(from_time, to_time, num_times, device=device)
    bottom_left, top_right = (bounds[0], bounds[2]), (bounds[1], bounds[3])
    H = int(abs(top_right[1] - bottom_left[1]) * density)
    W = int(abs(top_right[0] - bottom_left[0]) * density)
    D = len(bottom_left)
    T = len(time_grid)

    # Create spatial grid
    x_coords = torch.linspace(bottom_left[0], top_right[0], W, device=device)
    y_coords = torch.linspace(bottom_left[1], top_right[1], H, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [H, W]
    # Stack to [H, W, 2]
    grid = torch.stack([xx, yy], dim=-1)

    # Prepare output tensor
    tensor = torch.zeros((T, H, W, D), device=device)

    model.eval()
    with torch.no_grad():
        for t_idx, t_val in enumerate(time_grid):
            t_tensor = torch.tensor([t_val], device=device, dtype=torch.float32)
            # Broadcast t to match grid shape: [H, W, 1]
            t_broad = t_tensor.view(*([1] * (grid.dim() - 1)), 1)
            # Call model on full grid at once
            v = model(grid, t_broad)
            tensor[t_idx] = v

    return tensor

def get_velocity_field_tensor_3d(time_range: Tuple, num_times, bounds, density, model, device) -> Tensor:
    """
    :param time_range: (from, to)
    :param num_times: number of time steps
    :param bounds: bounding box ((x0, y0, z0), (x1, y1, z1))
    :param density: sampling density
    :param model: velocity model
    :param device: "cpu" or "cuda"
    :return: [T, D, H, W, 3] tensor
    """
    from_time, to_time = time_range
    time_grid = torch.linspace(from_time, to_time, num_times, device=device)

    bottom_left, top_right = bounds

    W = int(abs(top_right[0] - bottom_left[0]) * density)
    H = int(abs(top_right[1] - bottom_left[1]) * density)
    D = int(abs(top_right[2] - bottom_left[2]) * density)
    C = 3  # 3D vector field components
    T = len(time_grid)

    # Spatial grid
    x_coords = torch.linspace(bottom_left[0], top_right[0], W, device=device)
    y_coords = torch.linspace(bottom_left[1], top_right[1], H, device=device)
    z_coords = torch.linspace(bottom_left[2], top_right[2], D, device=device)

    zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')   # D, H, W

    grid = torch.stack([xx, yy, zz], dim=-1)  # [D, H, W, 3]

    tensor = torch.zeros((T, D, H, W, C), device=device)

    model.eval()
    with torch.no_grad():
        for t_idx, t_val in enumerate(time_grid):
            t_tensor = torch.tensor([t_val], device=device)
            t_broad = t_tensor.view(1, 1, 1, 1)
            v = model(grid, t_broad)   # returns [D, H, W, 3]
            tensor[t_idx] = v

    return tensor

def calculate_normalized_centers_ftd_dg_twice(d: int, mode_sep: float = 2.0):
    assert d >= 2
    delta = 4.0 / math.sqrt(d - 1)   # per-axis displacement
    a = delta / 2.0

    base_src = [-a] * (d - 1)
    base_tgt = [ a] * (d - 1)

    x0_c0 = base_src + [-mode_sep]
    x0_c1 = base_src + [ mode_sep]
    x1_c0 = base_tgt + [-mode_sep]
    x1_c1 = base_tgt + [ mode_sep]
    return x0_c0, x0_c1, x1_c0, x1_c1

