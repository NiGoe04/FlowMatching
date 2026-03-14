import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from torch.utils.data import DataLoader

from src.flow_matching.controller.cond_trainer import CondTrainer
from src.flow_matching.controller.lr_finder import LRFinder
from src.flow_matching.controller.utils import (
    get_velocity_field_tensor_2d,
    load_model_n_dim,
    store_model,
)
from src.flow_matching.model.coupling import Coupler
from src.flow_matching.model.distribution import Distribution
from src.flow_matching.model.losses import ConditionalFMLoss
from src.flow_matching.model.velocity_model_basic import SimpleVelocityModel
from src.flow_matching.shared.md_2d import PARAMS

# steering console
NAME = "2D_single_gauss_x_shift_snapshots"
FIND_LR = False
TRAIN_MODEL = False
SAVE_MODEL = False
GENERATE_SAMPLES = True
VISUALIZE_SNAPSHOTS = True

PLOT_BOUNDS = [-4, 4, -3, 3]
DIM = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "../../../../models"
MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "model_2D_single_gauss_twice_2026-03-14_12-19-54.pth")

# 0, 0.125, ..., 1.0 -> 9 time points
TIME_GRID = torch.linspace(0.0, 1.0, steps=9, device=DEVICE)


def plot_particles_over_time(x_t_samples, bounds, time_grid):
    """Create a 1 x T plot grid for particle snapshots."""
    num_times = len(time_grid)
    fig, axes = plt.subplots(1, num_times, figsize=(3 * num_times, 3.5), constrained_layout=True)

    if num_times == 1:
        axes = [axes]

    for idx, t in enumerate(time_grid):
        points = x_t_samples[idx].detach().cpu().numpy()

        ax = axes[idx]
        ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.65)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.set_title(f"t = {t.item():.3f}")

    plt.show()


def plot_velocity_field_over_time(field_tensor, bounds, time_grid):
    """Create a 1 x T plot grid for velocity field snapshots."""
    num_times = len(time_grid)

    field_np = field_tensor.detach().cpu().numpy()
    h, w = field_np.shape[1], field_np.shape[2]
    x = np.linspace(bounds[0], bounds[1], w)
    y = np.linspace(bounds[2], bounds[3], h)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    fig, axes = plt.subplots(1, num_times, figsize=(3 * num_times, 3.5), constrained_layout=True)

    if num_times == 1:
        axes = [axes]

    for idx, t in enumerate(time_grid):
        u = field_np[idx, :, :, 0]
        v = field_np[idx, :, :, 1]

        ax = axes[idx]
        ax.quiver(xx, yy, u, v)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.set_title(f"t = {t.item():.3f}")

    plt.show()


# data: single Gaussian from (-2, 0) to (2, 0)
variance_source = 0.1
variance_target = 0.1

x_0_center = [-2, 0]
x_1_center = [2, 0]

x_0_train = (
    Distribution(x_0_center, PARAMS["size_train_set"], device=DEVICE)
    .with_gaussian_noise(variance=variance_source)
    .tensor
)

x_1_train = (
    Distribution(x_1_center, PARAMS["size_train_set"], device=DEVICE)
    .with_gaussian_noise(variance=variance_target)
    .tensor
)

x_0_sample = (
    Distribution(x_0_center, PARAMS["amount_samples"], device=DEVICE)
    .with_gaussian_noise(variance=variance_source)
    .tensor
)

coupler = Coupler(x_0_train, x_1_train)
coupling = coupler.get_independent_coupling()
loader = DataLoader(coupling, PARAMS["batch_size"], shuffle=True)

# model
model = SimpleVelocityModel(device=DEVICE)
path = AffineProbPath(CondOTScheduler())
optimizer = torch.optim.Adam(model.parameters(), PARAMS["learning_rate"])
trainer = CondTrainer(
    model,
    optimizer,
    path,
    PARAMS["num_epochs"],
    PARAMS["num_trainer_val_samples"],
    device=DEVICE,
)

if FIND_LR:
    lr_finder = LRFinder(model, optimizer, path, ConditionalFMLoss(), device=DEVICE)
    lr_finder.range_test(loader, lr_start=1e-6, lr_end=1e-1, num_iters=100)
    lr_finder.plot()

if TRAIN_MODEL:
    trainer.training_loop(loader)

if SAVE_MODEL:
    MODEL_PATH = store_model(MODEL_SAVE_PATH, NAME, model)

if GENERATE_SAMPLES or VISUALIZE_SNAPSHOTS:
    model = load_model_n_dim(DIM, MODEL_PATH, device=DEVICE)
    solver = ODESolver(velocity_model=model)

if GENERATE_SAMPLES:
    x_1_sample = solver.sample(
        x_init=x_0_sample,
        method=PARAMS["solver_method"],
        step_size=1.0 / PARAMS["solver_steps"],
        time_grid=torch.tensor([0.0, 1.0], device=DEVICE),
    )
    print("Generated final samples:", x_1_sample.shape)

if VISUALIZE_SNAPSHOTS:
    x_t_samples = solver.sample(
        x_init=x_0_sample,
        method=PARAMS["solver_method"],
        step_size=1.0 / PARAMS["solver_steps"],
        return_intermediates=True,
        time_grid=TIME_GRID,
    )

    field_tensor = get_velocity_field_tensor_2d(
        time_range=(float(TIME_GRID[0]), float(TIME_GRID[-1])),
        num_times=len(TIME_GRID),
        bounds=PLOT_BOUNDS,
        density=PARAMS["field_density"],
        model=model,
        device=DEVICE,
    )

    plot_particles_over_time(x_t_samples, PLOT_BOUNDS, TIME_GRID)
    plot_velocity_field_over_time(field_tensor, PLOT_BOUNDS, TIME_GRID)