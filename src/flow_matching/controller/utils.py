from datetime import datetime

import torch
import os

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

def load_model_2d(model_class, model_path, device="cpu"):
    """
    Loads a model's state_dict from file.

    Args:
        model_class: the class of the model to instantiate, e.g. SimpleVelocityField
        model_path (str): path to the saved .pth file
        device (str): device to map the model to ("cpu" or "cuda")

    Returns:
        torch.nn.Module: the loaded model
    """
    # Create an instance of the model architecture
    model = model_class()

    # Load the saved weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Set model to evaluation mode by default
    model.to(device)
    model.eval()

    return model

def load_model_3d(model_class, model_path, device="cpu"):
    """
    Loads a model's state_dict from file.

    Args:
        model_class: the class of the model to instantiate, e.g. SimpleVelocityField
        model_path (str): path to the saved .pth file
        device (str): device to map the model to ("cpu" or "cuda")

    Returns:
        torch.nn.Module: the loaded model
    """
    # Create an instance of the model architecture
    model = model_class(dim=3)

    # Load the saved weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Set model to evaluation mode by default
    model.to(device)
    model.eval()

    return model