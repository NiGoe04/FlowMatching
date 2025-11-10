from typing import *

import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Slider


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


def visualize_mnist_samples(tensor: torch.Tensor, n_samples: int = 16, title: str = "MNIST Samples",
                            shuffle: bool = False):
    """
    Displays a grid of samples from a tensor of shape [N, 1, H, W].

    Args:
        tensor (torch.Tensor): Input tensor of shape [N, 1, H, W].
        n_samples (int): How many images to display.
        title (str): Title for the plot.
        shuffle (bool): Whether to shuffle the samples before displaying.
    """
    # noinspection PyPep8Naming
    N, C, H, W = tensor.shape
    if C != 1:
        raise ValueError(f"Expected single-channel tensor, got {C} channels")

    n_samples = min(n_samples, N)
    indices = torch.randperm(N) if shuffle else torch.arange(N)
    indices = indices[:n_samples]

    n_cols = int(n_samples ** 0.5)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for i in range(n_samples):
        axes[i].imshow(tensor[indices[i], 0].cpu().numpy(), cmap="gray")
        axes[i].axis("off")

    for i in range(n_samples, len(axes)):
        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_multi_slider_ndim(tensors: Sequence[torch.Tensor],
                                time_grid: torch.Tensor):
    """
    Visualize n-dimensional ODE solutions with a slider over time.
    Supports 2D or 3D data (D=2 or D=3).

    Args:
        tensors (Sequence[torch.Tensor]): List of tensors, each shape [N, D].
        time_grid (torch.Tensor): Tensor of times, shape [len(ode_solutions)].
    """
    assert len(tensors) == len(time_grid), "Number of solutions must match number of times."

    pos_slider = (0.2, 0.02, 0.6, 0.03)
    plt.switch_backend("tkagg")

    # Determine dimensionality
    N, D = tensors[0].shape
    if D not in [2, 3]:
        raise ValueError("Only 2D or 3D data is supported for visualization.")

    # Move to CPU for plotting
    tensors = [tensor.cpu() for tensor in tensors]

    # Initial plot
    fig = plt.figure(figsize=(6, 6))
    if D == 3:
        ax = fig.add_subplot(111, projection='3d')
        all_points = torch.cat(tensors, dim=0)
        ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
        ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
        ax.set_zlim(all_points[:, 2].min(), all_points[:, 2].max())
        scatter = ax.scatter(*tensors[0].T)
    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(*tensors[0].T)

    ax.set_title(f"t = {time_grid[0].item():.3f}")

    # Slider
    ax_slider = plt.axes(pos_slider)
    slider = Slider(ax_slider, 't', float(time_grid.min()), float(time_grid.max()), valinit=float(time_grid[0]))

    # noinspection PyUnusedLocal
    def update(val):
        nonlocal scatter
        idx = (torch.abs(time_grid - slider.val)).argmin().item()
        points = tensors[idx]

        if D == 2:
            scatter.set_offsets(points)
        else:  # 3D: need to clear and redraw
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
            ax.cla()
            scatter = ax.scatter(*points.T)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

        ax.set_title(f"t = {time_grid[idx].item():.3f}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def visualize_multi_slider_mnist(tensors: Sequence[torch.Tensor],
                                 t_values: torch.Tensor,
                                 num_samples: int = 16,
                                 shuffle: bool = True,
                                 title_prefix: str = "MNIST Samples"):
    """
    Visualize MNIST ODE solutions with a slider over time.
    Shows a grid of images for each time step.

    Args:
        tensors (Sequence[torch.Tensor]): List of tensors, each shape [N, 1, H, W].
        t_values (torch.Tensor): Tensor of times, shape [len(tensors)].
        num_samples (int): How many images to display in the grid.
        shuffle (bool): Whether to randomly select images.
        title_prefix (str): Prefix for the plot title.
    """
    assert len(tensors) == len(t_values), "Number of solutions must match number of times."

    pos_layout = (0, 0.05, 1, 0.95)
    pos_slider = (0.2, 0.02, 0.6, 0.03)

    plt.switch_backend("tkagg")

    # need to use CPU for plotting
    tensors = [tensor.cpu() for tensor in tensors]
    N, C, H, W = tensors[0].shape
    if C != 1:
        raise ValueError(f"Expected single-channel tensors, got {C} channels")

    num_samples = min(num_samples, N)
    indices = torch.randperm(N)[:num_samples] if shuffle else torch.arange(num_samples)

    n_cols = int(num_samples ** 0.5)
    n_rows = (num_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    # Initial display
    for i in range(num_samples):
        axes[i].imshow(tensors[0][indices[i], 0].numpy(), cmap="gray")
        axes[i].axis("off")

    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"{title_prefix} at t = {t_values[0].item():.3f}")
    plt.tight_layout(rect=pos_layout)

    # Slider axes
    ax_slider = plt.axes(pos_slider)
    slider = Slider(ax_slider, 't', float(t_values.min()), float(t_values.max()), valinit=float(t_values[0]))

    # noinspection PyUnusedLocal
    def update(val):
        t_val = slider.val
        idx = (torch.abs(t_values - t_val)).argmin().item()
        for j in range(num_samples):
            axes[j].imshow(tensors[idx][indices[j], 0].numpy(), cmap="gray")
        plt.suptitle(f"{title_prefix} at t = {t_values[idx].item():.3f}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
