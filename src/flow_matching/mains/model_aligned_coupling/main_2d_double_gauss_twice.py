import os

import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from torch.utils.data import DataLoader

from src.flow_matching.controller.cond_trainer import CondTrainer, CondTrainerMAC
from src.flow_matching.controller.lr_finder import LRFinder
from src.flow_matching.controller.utils import store_model, load_model_n_dim
from src.flow_matching.model.coupling import Coupler
from src.flow_matching.model.distribution import Distribution
from src.flow_matching.model.losses import ConditionalFMLoss
from src.flow_matching.model.velocity_model_basic import SimpleVelocityModel
from src.flow_matching.shared.md_2d import PARAMS
from src.flow_matching.shared.md_mac import PARAMS_MAC
from src.flow_matching.view.utils import plot_tensor_2d, visualize_multi_slider_ndim, visualize_velocity_field_2d

# steering console
NAME = "2D_double_gauss_twice_mac"
FIND_LR = False
PLOT_TRAIN_DATA = False
TRAIN_MODEL = False
SAVE_MODEL = False
GENERATE_SAMPLES = True
VISUALIZE_TIME = True
VISUALIZE_FIELD = True

PLOT_BOUNDS = [-4, 4, -4, 4]
DIM = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "../../../../models"

# data
variance_source = 0.1
variance_target = 0.1

x_0_dist_center_0 = [-2, -2]
x_0_dist_center_1 = [-2, 2]
x_1_dist_center_0 = [2, -2]
x_1_dist_center_1 = [2, 2]

x_0_dist_0 = (Distribution(x_0_dist_center_0, int(PARAMS["size_train_set"] / 2), device=DEVICE)
              .with_gaussian_noise(variance=variance_source))

x_0_dist_1 = (Distribution(x_0_dist_center_1, int(PARAMS["size_train_set"] / 2), device=DEVICE)
              .with_gaussian_noise(variance=variance_source))

x_1_dist_0 = (Distribution(x_1_dist_center_0, int(PARAMS["size_train_set"] / 2), device=DEVICE)
              .with_gaussian_noise(variance=variance_target))

x_1_dist_1 = (Distribution(x_1_dist_center_1, int(PARAMS["size_train_set"] / 2), device=DEVICE)
              .with_gaussian_noise(variance=variance_target))

x_0_dist = x_0_dist_0.merged_with(x_0_dist_1)
x_1_dist = x_1_dist_0.merged_with(x_1_dist_1)

x_0_train = x_0_dist.tensor
x_1_train = x_1_dist.tensor

x_0_dist_sample_0 = (Distribution(x_0_dist_center_0, int(PARAMS["amount_samples"] / 2), device=DEVICE)
                     .with_gaussian_noise(variance=variance_source))

x_0_dist_sample_1 = (Distribution(x_0_dist_center_1, int(PARAMS["amount_samples"] / 2), device=DEVICE)
                     .with_gaussian_noise(variance=variance_source))

x_0_sample = x_0_dist_sample_0.merged_with(x_0_dist_sample_1).tensor

if PLOT_TRAIN_DATA:
    plot_tensor_2d(x_0_train)
    plot_tensor_2d(x_1_train)

coupler = Coupler(x_0_train, x_1_train)
coupling = coupler.get_independent_coupling()
loader = DataLoader(
    coupling,
    PARAMS["batch_size"],
    shuffle=True,
)

# model
model = SimpleVelocityModel(device=DEVICE)
path = AffineProbPath(CondOTScheduler())
optimizer = torch.optim.Adam(model.parameters(), PARAMS["learning_rate"])
trainer_warmup = CondTrainer(model, optimizer, path, 1, PARAMS["num_trainer_val_samples"], device=DEVICE)
trainer_mac = CondTrainerMAC(model, optimizer, path, PARAMS_MAC["num_epochs"], PARAMS["num_trainer_val_samples"],
                             PARAMS_MAC["top_k_percentage"], PARAMS_MAC["mac_reg_coefficient"], device=DEVICE)
model_path = os.path.join(MODEL_SAVE_PATH, "model_2D_double_gauss_twice_mac_2026-01-29_17-12-25.pth")

if FIND_LR:
    lr_finder = LRFinder(model, optimizer, path, ConditionalFMLoss(), device=DEVICE)
    lr_finder.range_test(loader, lr_start=1e-6, lr_end=1e-1, num_iters=100)
    lr_finder.plot()

# training
if TRAIN_MODEL:
    trainer_warmup.training_loop(loader)
    trainer_mac.training_loop(loader)

if SAVE_MODEL:
    # noinspection PyRedeclaration
    model_path = store_model(MODEL_SAVE_PATH, NAME, model)

if GENERATE_SAMPLES:
    model = load_model_n_dim(DIM, model_path, device=DEVICE)
    solver = ODESolver(velocity_model=model)
    x_1_sample = solver.sample(x_init=x_0_sample, method=PARAMS["solver_method"],
                               step_size=1.0 / PARAMS["solver_steps"],
                               time_grid=torch.Tensor([0.0, PARAMS["t_end"]]))
    plot_tensor_2d(x_1_sample, params=None)

if VISUALIZE_TIME:
    model = load_model_n_dim(DIM, model_path, device=DEVICE)
    solver = ODESolver(velocity_model=model)
    time_grid = torch.linspace(0.0, PARAMS["t_end"], steps=PARAMS["num_times_to_visualize"], device=DEVICE)
    x_1_samples = solver.sample(x_init=x_0_sample, method=PARAMS["solver_method"],
                                step_size=1.0 / PARAMS["solver_steps"],
                                return_intermediates=True, time_grid=time_grid)
    visualize_multi_slider_ndim(x_1_samples, time_grid, bounds=PLOT_BOUNDS)

if VISUALIZE_FIELD:
    model = load_model_n_dim(DIM, model_path, device=DEVICE)
    time_range = (0.0, PARAMS["t_end"])
    visualize_velocity_field_2d(time_range, PARAMS["num_times_to_visualize"], PLOT_BOUNDS,
                                PARAMS["field_density"], model, DEVICE)
