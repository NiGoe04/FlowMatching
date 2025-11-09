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
from src.flow_matching.model.losses import ConditionalFMLoss
from src.flow_matching.model.velocity_model_basic import SimpleVelocityModel
from src.flow_matching.view.utils import plot_tensor_2d, visualize_multi_data_slider_ndim

# steering console
NAME = "2D"
FIND_LR = False
TRAIN_MODEL = False
SAVE_MODEL = False
GENERATE_SAMPLES = False
VISUALIZE_TIME = True

DIM = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "../../../models"
# hyperparams
PARAMS = {
    "num_epochs": 15,
    "batch_size": 256,
    "learning_rate": 1e-2,
    "size_train_set": 200000,
    "amount_samples": 500,
    "solver_steps": 100,
    "num_times_to_visualize": 150,
    "t_end": 1.0,   # just for time visualization
    "solver_method": 'midpoint'
}

# data
x_0_train = torch.randn(PARAMS["size_train_set"], DIM, device=DEVICE)
x_1_train = Tensor(make_moons(PARAMS["size_train_set"], noise=0.05)[0]).to(DEVICE)
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
model_path = os.path.join(MODEL_SAVE_PATH, "model_2D_2025-11-09_14-28-07.pth")

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
    x_0_sample = torch.randn(PARAMS["amount_samples"], DIM, device=DEVICE)
    x_1_sample = solver.sample(x_init=x_0_sample, method=PARAMS["solver_method"], step_size=1.0 / PARAMS["solver_steps"])
    plot_tensor_2d(x_1_sample, params=PARAMS)

if VISUALIZE_TIME:
    model = load_model_n_dim(DIM, model_path, device=DEVICE)
    solver = ODESolver(velocity_model=model)
    x_0_sample = torch.randn(PARAMS["amount_samples"], DIM, device=DEVICE)
    time_grid = torch.linspace(0.0, PARAMS["t_end"], steps=PARAMS["num_times_to_visualize"], device=DEVICE)
    x_1_samples = solver.sample(x_init=x_0_sample, method=PARAMS["solver_method"], step_size=1.0 / PARAMS["solver_steps"],
                                return_intermediates=True, time_grid=time_grid)
    visualize_multi_data_slider_ndim(x_1_samples, time_grid)
