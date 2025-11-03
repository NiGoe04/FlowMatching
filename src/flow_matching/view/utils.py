import matplotlib.pyplot as plt
from typing import *
import torch
from mpl_toolkits import mplot3d

def plot_tensor_2d(points: torch.Tensor,
                   title: str = "2D Scatter Plot",
                   params: Optional[Dict[str, Any]] = None):
    """
    Plots a 2D scatter plot from a tensor of shape [n, 2], optionally annotating it with parameters.

    Args:
        points (torch.Tensor): Tensor of shape [n, 2].
        title (str): Title of the plot.
        params (dict, optional): Dictionary of parameters to display on the plot.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected tensor of shape [n, 2], got {points.shape}")

    x = points[:, 0].cpu().numpy()
    y = points[:, 1].cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)

    if params:
        # Parameter-Text to the upper left
        param_text = "\n".join([f"{k}: {v}" for k, v in params.items()])
        plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='left',
                 fontsize=9, bbox=dict(facecolor='white', alpha=0.6))

    plt.show()

def plot_tensor_3d(points: torch.Tensor, title: str = "3D Scatter Plot", params: dict = None):
    """
    Plots a 3D scatter plot from a tensor of shape [n, 3].

    Args:
        points (torch.Tensor): Tensor of shape [n, 3].
        title (str): Title of the plot.
        params (dict, optional): Dictionary of parameters to display on the plot. Default is None.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected tensor of shape [n, 3], got {points.shape}")

    plt.switch_backend("tkagg")

    x = points[:, 0].cpu().numpy()
    y = points[:, 1].cpu().numpy()
    z = points[:, 2].cpu().numpy()

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=10, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("x₃")

    if params is not None:
        param_text = "\n".join(f"{k}: {v}" for k, v in params.items())
        ax.text2D(0.05, 0.95, param_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')

    plt.show()
