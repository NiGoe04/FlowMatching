import os

import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from sklearn.datasets import make_moons
from torch import Tensor
from torch.utils.data import DataLoader

from src.flow_matching.controller.cond_trainer import CondTrainer
from src.flow_matching.controller.lr_finder import LRFinder
from src.flow_matching.controller.utils import store_model, load_model_n_dim
from src.flow_matching.model.coupling import Coupler
from src.flow_matching.model.distribution import Distribution
from src.flow_matching.model.losses import ConditionalFMLoss
from src.flow_matching.model.velocity_model_basic import SimpleVelocityModel
from src.flow_matching.shared.data_2d import PARAMS, noise_bound_target
from src.flow_matching.view.utils import plot_tensor_2d, visualize_multi_slider_ndim, visualize_velocity_field_2d

# steering console
NAME = "2D_bounded_uni"
FIND_LR = False
PLOT_TRAIN_DATA = False
TRAIN_MODEL = False
SAVE_MODEL = False
GENERATE_SAMPLES = True
VISUALIZE_TIME = True
VISUALIZE_FIELD = True

DIM = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "../../../../models"

# data
noise_bound_source = 2

x_0_dist_center = [0.6, 0.3]
x_0_train = (Distribution.get_uni_distribution(x_0_dist_center, PARAMS["size_train_set"], device=DEVICE)
             .with_2d_uniform_noise(noise_bound_source)).tensor

x_1_train = (Distribution(Tensor(make_moons(PARAMS["size_train_set"], noise=0.00)[0]), device=DEVICE)
             .with_2d_uniform_noise(noise_bound_target)).tensor

x_0_sample = (Distribution.get_uni_distribution(x_0_dist_center, PARAMS["amount_samples"], device=DEVICE)
              .with_2d_uniform_noise(noise_bound_source)).tensor

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
trainer = CondTrainer(model, optimizer, path, PARAMS["num_epochs"], device=DEVICE)
model_path = os.path.join(MODEL_SAVE_PATH, "model_2D_bounded_uni_2025-11-17_15-06-41.pth")

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
    visualize_multi_slider_ndim(x_1_samples, time_grid)

if VISUALIZE_FIELD:
    model = load_model_n_dim(DIM, model_path, device=DEVICE)
    time_range = (0.0, PARAMS["t_end"])
    visualize_velocity_field_2d(time_range, PARAMS["num_times_to_visualize"], PARAMS["field_bound"],
                                PARAMS["field_density"], model, DEVICE)
