import os

import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from torch.utils.data import DataLoader

from src.flow_matching.controller.cond_trainer import CondTrainer
from src.flow_matching.controller.lr_finder import LRFinder
from src.flow_matching.controller.utils import store_model, load_mnist_tensor, load_model_unet
from src.flow_matching.model.coupling import Coupler
from src.flow_matching.model.losses import ConditionalFMLoss
from src.flow_matching.model.velocity_model_unet import UnetVelocityModel
from src.flow_matching.view.utils import visualize_mnist_samples, visualize_multi_slider_mnist

# steering console
SET_TYPE = "MNIST_F"  # "MNIST_F" -> Fashion-MNIST, "MNIST_N" -> Standard MNIST
FIND_LR = False
TRAIN_MODEL = False
SAVE_MODEL = False
GENERATE_SAMPLES = False
VISUALIZE_TIME = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "../../../../models"

# hyperparams
PARAMS = {
    "num_epochs": 15,
    "batch_size": 32,
    "dropout_rate_model": 0.00,
    "learning_rate": 1e-3,
    "size_train_set": 60000,
    "amount_samples": 4,
    "solver_steps": 200,
    "num_times_to_visualize": 30,
    "t_end": 1.05,
    "solver_method": 'midpoint'
}

# data
x_1_train = load_mnist_tensor(PARAMS["size_train_set"], device=DEVICE, set_type=SET_TYPE)
x_0_train = torch.randn_like(x_1_train, device=DEVICE)
coupler = Coupler(x_0_train, x_1_train)
coupling = coupler.get_independent_coupling()
loader = DataLoader(
    coupling,
    PARAMS["batch_size"],
    shuffle=True,
)

# model
model = UnetVelocityModel(dropout_rate=PARAMS["dropout_rate_model"], device=DEVICE)
path = AffineProbPath(CondOTScheduler())
optimizer = torch.optim.Adam(model.parameters(), PARAMS["learning_rate"])
trainer = CondTrainer(model, optimizer, path, PARAMS["num_epochs"], PARAMS["num_trainer_val_samples"], device=DEVICE)
# model_MNIST_F_2025-11-05_16-02-20.pth
# model_MNIST_N_2025-11-05_21-15-20.pth
model_path = os.path.join(MODEL_SAVE_PATH, "model_MNIST_F_2025-11-05_16-02-20.pth")

# execution
if FIND_LR:
    lr_finder = LRFinder(model, optimizer, path, ConditionalFMLoss(), device=DEVICE)
    lr_finder.range_test(loader, lr_start=1e-6, lr_end=1.0, num_iters=100)
    lr_finder.plot()

if TRAIN_MODEL:
    trainer.training_loop(loader)

if SAVE_MODEL:
    # noinspection PyRedeclaration
    model_path = store_model(MODEL_SAVE_PATH, SET_TYPE, model)

if GENERATE_SAMPLES:
    model = load_model_unet(model_path, PARAMS["dropout_rate_model"], DEVICE)
    solver = ODESolver(velocity_model=model)
    x_0_sample = torch.randn(PARAMS["amount_samples"], *x_0_train.shape[1:], device=DEVICE)
    x_1_sample = solver.sample(x_init=x_0_sample, method=PARAMS["solver_method"],
                               step_size=1.0 / PARAMS["solver_steps"],
                               time_grid=torch.Tensor([0.0, PARAMS["t_end"]]))
    visualize_mnist_samples(x_1_sample, n_samples=PARAMS["amount_samples"])

if VISUALIZE_TIME:
    model = load_model_unet(model_path, PARAMS["dropout_rate_model"], DEVICE)
    solver = ODESolver(velocity_model=model)
    x_0_sample = torch.randn(PARAMS["amount_samples"], *x_0_train.shape[1:], device=DEVICE)
    time_grid = torch.linspace(0.0, PARAMS["t_end"], steps=PARAMS["num_times_to_visualize"], device=DEVICE)
    x_1_sample = solver.sample(x_init=x_0_sample, method=PARAMS["solver_method"],
                               step_size=1.0 / PARAMS["solver_steps"],
                               return_intermediates=True, time_grid=time_grid)
    visualize_multi_slider_mnist(x_1_sample, time_grid, num_samples=PARAMS["amount_samples"])
