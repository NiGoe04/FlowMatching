from typing import *

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required)

from src.flow_matching.controller.utils import get_velocity_field_tensor_2d, get_velocity_field_tensor_3d


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

    # Keep an equal data aspect ratio so isotropic clouds are not visually flattened.
    x_range = max(float(np.max(x) - np.min(x)), 1e-8)
    y_range = max(float(np.max(y) - np.min(y)), 1e-8)
    z_range = max(float(np.max(z) - np.min(z)), 1e-8)
    ax.set_box_aspect((x_range, y_range, z_range))

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


def visualize_rgb_samples(
    tensor: torch.Tensor,
    start_idx: int,
    end_idx: int,
    title: str = "RGB samples",
):
    """
    Displays RGB images from a tensor of shape [N, 3, H, W] for indices in [start_idx, end_idx].
    """
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(f"Expected tensor with shape [N, 3, H, W], got {tuple(tensor.shape)}")
    if start_idx < 0 or end_idx < start_idx or end_idx >= tensor.shape[0]:
        raise ValueError(
            f"Invalid range [{start_idx}, {end_idx}] for dataset of size {tensor.shape[0]}"
        )

    num_samples = end_idx - start_idx + 1
    n_cols = int(np.ceil(np.sqrt(num_samples)))
    n_rows = int(np.ceil(num_samples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    axes = np.array(axes).reshape(-1)

    for plot_idx, sample_idx in enumerate(range(start_idx, end_idx + 1)):
        img = tensor[sample_idx].detach().cpu()
        img = img - img.min()
        img = img / (img.max() + 1e-8)
        axes[plot_idx].imshow(img.permute(1, 2, 0).numpy())
        axes[plot_idx].set_title(str(sample_idx))
        axes[plot_idx].axis("off")

    for plot_idx in range(num_samples, len(axes)):
        axes[plot_idx].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_rgb_trajectories(
    tensors: Sequence[torch.Tensor],
    time_grid: torch.Tensor,
    sample_indices: Sequence[int],
):
    """
    Visualize selected RGB samples over time as a static panel.
    Rows correspond to samples, columns to time points.
    """
    if len(tensors) != len(time_grid):
        raise ValueError("Number of tensors must match number of time values.")

    num_rows = len(sample_indices)
    num_cols = len(tensors)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2.2 * num_cols, 2.2 * num_rows))
    axes = np.array(axes).reshape(num_rows, num_cols)

    for row, sample_idx in enumerate(sample_indices):
        for col, x_t in enumerate(tensors):
            img = x_t[sample_idx].detach().cpu()
            img = img - img.min()
            img = img / (img.max() + 1e-8)
            axes[row, col].imshow(img.permute(1, 2, 0).numpy())
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"t={time_grid[col].item():.2f}")

    plt.tight_layout()
    plt.show()

def visualize_multi_slider_ndim(tensors: Sequence[torch.Tensor],
                                time_grid: torch.Tensor,
                                bounds: Optional[Sequence[float]] = None):
    """
    Visualize n-dimensional ODE solutions with a slider over time.
    Supports 2D or 3D data (D=2 or D=3).

    Args:
        tensors (Sequence[torch.Tensor]): List of tensors, each shape [N, D].
        time_grid (torch.Tensor): Times corresponding to each tensor.
        bounds (list or tuple, optional):
            2D: [xmin, xmax, ymin, ymax]
            3D: [xmin, xmax, ymin, ymax, zmin, zmax]
    """
    assert len(tensors) == len(time_grid), "Number of solutions must match number of times."

    pos_slider = (0.2, 0.02, 0.6, 0.03)
    plt.switch_backend("tkagg")

    # Determine dimensionality
    N, D = tensors[0].shape
    if D not in [2, 3]:
        raise ValueError("Only 2D or 3D data is supported for visualization.")

    # Validate bounds
    if bounds is not None:
        if D == 2 and len(bounds) != 4:
            raise ValueError("2D bounds must be [xmin, xmax, ymin, ymax].")
        if D == 3 and len(bounds) != 6:
            raise ValueError("3D bounds must be [xmin, xmax, ymin, ymax, zmin, zmax].")

    # Move to CPU for plotting
    tensors = [tensor.cpu() for tensor in tensors]

    # Initial plot
    fig = plt.figure(figsize=(6, 6))

    if D == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(*tensors[0].T)

        # Set frozen axes
        if bounds is not None:
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
            ax.set_zlim(bounds[4], bounds[5])
        else:
            all_points = torch.cat(tensors, dim=0)
            ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
            ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
            ax.set_zlim(all_points[:, 2].min(), all_points[:, 2].max())

    else:  # 2D
        ax = fig.add_subplot(111)
        scatter = ax.scatter(*tensors[0].T)

        if bounds is not None:
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
        else:
            all_points = torch.cat(tensors, dim=0)
            ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
            ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())

    ax.set_title(f"t = {time_grid[0].item():.3f}")

    # Slider
    ax_slider = plt.axes(pos_slider)
    slider = Slider(ax_slider, 't', float(time_grid.min()), float(time_grid.max()),
                    valinit=float(time_grid[0]))

    # Update callback
    def update(val):
        nonlocal scatter
        idx = (torch.abs(time_grid - slider.val)).argmin().item()
        points = tensors[idx]

        if D == 2:
            scatter.set_offsets(points)
        else:
            # For 3D: clear and redraw, but reapply bounds (frozen axes)
            ax.cla()
            scatter = ax.scatter(*points.T)

            if bounds is not None:
                ax.set_xlim(bounds[0], bounds[1])
                ax.set_ylim(bounds[2], bounds[3])
                ax.set_zlim(bounds[4], bounds[5])
            else:
                # Reapply automatically computed limits
                all_points = torch.cat(tensors, dim=0)
                ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
                ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
                ax.set_zlim(all_points[:, 2].min(), all_points[:, 2].max())

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

def _visualize_velocity_field_2d(time_range: Tuple, num_times, bounds, field_tensor):
    """
    :param time_range: time range (from, to)
    :param num_times: amount times
    :param bounds: bounding box size (bottom_left, top_right)
    :param field_tensor: the field to visualize
    """
    plt.switch_backend("tkagg")  # use GUI backend
    pos_slider = (0.25, 0.1, 0.5, 0.03)

    from_time, to_time = time_range
    bottom_left, top_right = (bounds[0], bounds[2]), (bounds[1], bounds[3])
    H, W = field_tensor.shape[1], field_tensor.shape[2]

    # Create coordinate grid
    x = torch.linspace(bottom_left[0], top_right[0], W)
    y = torch.linspace(bottom_left[1], top_right[1], H)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    xx, yy = xx.numpy(), yy.numpy()

    field_tensor_np = field_tensor.cpu().numpy()  # convert to numpy

    # Initial plot
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.25)

    u = field_tensor_np[0, :, :, 0]
    v = field_tensor_np[0, :, :, 1]
    quiver = ax.quiver(xx, yy, u, v)
    ax.set_title(f"Velocity field at time {from_time:.2f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(bottom_left[0], top_right[0])
    ax.set_ylim(bottom_left[1], top_right[1])
    ax.set_aspect('equal')

    # Slider axis
    ax_slider = plt.axes(pos_slider)
    slider = Slider(ax_slider, 'Time step', 0, num_times - 1, valinit=0, valstep=1)

    # Update function
    def update(val):
        t_index = int(val)
        t_real = from_time + t_index / (num_times - 1) * (to_time - from_time)
        u = field_tensor_np[t_index, :, :, 0]
        v = field_tensor_np[t_index, :, :, 1]
        quiver.set_UVC(u, v)
        ax.set_title(f"Velocity field at time {t_real:.2f}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def visualize_velocity_field_2d(time_range: Tuple, num_times, bounds, density, model, device):
    field_tensor = get_velocity_field_tensor_2d(time_range, num_times, bounds, density, model, device)
    _visualize_velocity_field_2d(time_range, num_times, bounds, field_tensor)

def _visualize_velocity_field_3d(time_range: Tuple, num_times, bounds, field_tensor):
    """
    3D vector field visualization with a single slider for time.
    :param time_range: (from, to)
    :param num_times: number of time samples
    :param bounds: ((x0,y0,z0), (x1,y1,z1))
    :param field_tensor: [T, D, H, W, 3]
    """
    plt.switch_backend("tkagg")
    pos_slider = (0.25, 0.1, 0.5, 0.03)

    from_time, to_time = time_range
    bottom_left, top_right = bounds

    T, D, H, W, _ = field_tensor.shape
    field_np = field_tensor.cpu().numpy()

    # Coordinate grid
    x = np.linspace(bottom_left[0], top_right[0], W)
    y = np.linspace(bottom_left[1], top_right[1], H)
    z = np.linspace(bottom_left[2], top_right[2], D)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')

    # Flattened coordinates for quiver
    Xf = xx.flatten()
    Yf = yy.flatten()
    Zf = zz.flatten()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)

    # Initial quiver (time = 0)
    Uf = field_np[0, :, :, :, 0].flatten()
    Vf = field_np[0, :, :, :, 1].flatten()
    Wf = field_np[0, :, :, :, 2].flatten()

    quiver_container = {"obj": ax.quiver(Xf, Yf, Zf, Uf, Vf, Wf, length=0.15, normalize=False)}

    ax.set_title(f"3D velocity field at time {from_time:.2f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Time slider
    ax_slider = plt.axes(pos_slider)
    slider = Slider(ax_slider, 'Time', 0, num_times - 1, valinit=0, valstep=1)

    def update(val):
        t_idx = int(slider.val)
        t_real = from_time + t_idx / (num_times - 1) * (to_time - from_time)

        old = quiver_container["obj"]
        if old is not None and old in ax.collections:
            old.remove()

        Uf = field_np[t_idx, :, :, :, 0].flatten()
        Vf = field_np[t_idx, :, :, :, 1].flatten()
        Wf = field_np[t_idx, :, :, :, 2].flatten()

        quiver_container["obj"] = ax.quiver(Xf, Yf, Zf, Uf, Vf, Wf, length=0.15, normalize=False)

        ax.set_title(f"3D velocity field at time {t_real:.2f}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def visualize_velocity_field_3d(time_range, num_times, bounds, density, model, device):
    field_tensor = get_velocity_field_tensor_3d(time_range, num_times, bounds, density, model, device)
    _visualize_velocity_field_3d(time_range, num_times, bounds, field_tensor)
