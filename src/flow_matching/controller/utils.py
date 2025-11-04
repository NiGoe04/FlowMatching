from datetime import datetime

import torch
import os

from src.flow_matching.model.velocity_model_basic import SimpleVelocityModel


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