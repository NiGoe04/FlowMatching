import os

import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from torch.utils.data import DataLoader

from src.experiments.scenarios import get_scenario
from src.flow_matching.controller.cond_trainer import CondTrainerBatchOT
from src.flow_matching.controller.lr_finder import LRFinder
from src.flow_matching.controller.utils import store_model, load_model_n_dim
from src.flow_matching.model.coupling import Coupler
from src.flow_matching.model.losses import ConditionalFMLoss
from src.flow_matching.model.velocity_model_basic import SimpleVelocityModel
from src.flow_matching.shared.md_2d import PARAMS
from src.flow_matching.view.utils import (
    plot_tensor_2d,
    visualize_multi_slider_ndim,
    visualize_velocity_field_2d,
)

# steering console
NAME = "2D_gaussian_mix_diff_var_1_ot"
FIND_LR = True
PLOT_TRAIN_DATA = True
TRAIN_MODEL = True
SAVE_MODEL = True
GENERATE_SAMPLES = True
VISUALIZE_TIME = True
VISUALIZE_FIELD = True

# circles are centered at radius 4 (x0) and radius 8 (x1)
PLOT_BOUNDS = [-10, 10, -10, 10]
DIM = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "../../../../models"

SCENARIO = "gaussian_mix_diff_var_1"

# data (via GaussianMixtureDistribution from scenario helpers)
gmd_x0, gmd_x1, w2_sq_pre_calc = get_scenario(SCENARIO, DIM, DEVICE)

# training tensors
x_0_train = gmd_x0.sample(PARAMS["size_train_set"])
x_1_train = gmd_x1.sample(PARAMS["size_train_set"])

# sampling tensor (for generation / visualization)
x_0_sample = gmd_x0.sample(PARAMS["amount_samples"])

if PLOT_TRAIN_DATA:
    plot_tensor_2d(x_0_train)
    plot_tensor_2d(x_1_train)

# coupling + loader (same pattern as your double-gauss script)
coupler = Coupler(x_0_train, x_1_train)
coupling = coupler.get_independent_coupling()
loader = DataLoader(
    coupling,
    PARAMS["batch_size"],
    shuffle=True,
)

# model + trainer
model = SimpleVelocityModel(device=DEVICE)
path = AffineProbPath(CondOTScheduler())
optimizer = torch.optim.Adam(model.parameters(), PARAMS["learning_rate"])
trainer = CondTrainerBatchOT(
    model,
    optimizer,
    path,
    PARAMS["num_epochs"],
    PARAMS["num_trainer_val_samples"],
    device=DEVICE,
)

# set a default model path (replace with your actual checkpoint if you want)
model_path = os.path.join(MODEL_SAVE_PATH, "model_2D_gaussian_circles.pth")

if FIND_LR:
    lr_finder = LRFinder(model, optimizer, path, ConditionalFMLoss(), device=DEVICE)
    lr_finder.range_test(loader, lr_start=1e-6, lr_end=1e-1, num_iters=100)
    lr_finder.plot()

# training
if TRAIN_MODEL:
    trainer.training_loop(loader)

if SAVE_MODEL:
    # noinspection PyRedeclaration
    model_path = store_model(MODEL_SAVE_PATH, NAME, model)

# generate samples at t=1
if GENERATE_SAMPLES:
    model = load_model_n_dim(DIM, model_path, device=DEVICE)
    solver = ODESolver(velocity_model=model)
    x_1_sample = solver.sample(
        x_init=x_0_sample,
        method=PARAMS["solver_method"],
        step_size=1.0 / PARAMS["solver_steps"],
        time_grid=torch.Tensor([0.0, PARAMS["t_end"]]),
    )
    plot_tensor_2d(x_1_sample, params=None)

# visualize intermediate times
if VISUALIZE_TIME:
    model = load_model_n_dim(DIM, model_path, device=DEVICE)
    solver = ODESolver(velocity_model=model)
    time_grid = torch.linspace(0.0, PARAMS["t_end"], steps=PARAMS["num_times_to_visualize"], device=DEVICE)
    x_1_samples = solver.sample(
        x_init=x_0_sample,
        method=PARAMS["solver_method"],
        step_size=1.0 / PARAMS["solver_steps"],
        return_intermediates=True,
        time_grid=time_grid,
    )
    visualize_multi_slider_ndim(x_1_samples, time_grid, bounds=PLOT_BOUNDS)

# visualize velocity field
if VISUALIZE_FIELD:
    model = load_model_n_dim(DIM, model_path, device=DEVICE)
    time_range = (0.0, PARAMS["t_end"])
    visualize_velocity_field_2d(
        time_range,
        PARAMS["num_times_to_visualize"],
        PLOT_BOUNDS,
        PARAMS["field_density"],
        model,
        DEVICE,
    )
