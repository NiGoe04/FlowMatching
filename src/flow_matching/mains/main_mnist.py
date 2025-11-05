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
from src.flow_matching.view.utils import visualize_mnist_samples

# steering console
SET_TYPE = "MNIST_F" # "MNIST_F" -> Fashion-MNIST, "MNIST_N" -> Standard MNIST
FIND_LR = False
TRAIN_MODEL = True
SAVE_MODEL = True
SAMPLE_FROM_MODEL = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "../../../models"

# hyperparams
PARAMS = {
    "num_epochs": 15,
    "batch_size": 32,
    "dropout_rate_model": 0.00,
    "learning_rate": 1e-3,
    "size_train_set": 60000,
    "amount_samples": 16,
    "solver_steps": 200,
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
trainer = CondTrainer(model, optimizer, path, PARAMS["num_epochs"], DEVICE)
model_path = os.path.join(MODEL_SAVE_PATH, "model_MNIST_F_2025-11-05_13-38-27.pth")

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

if SAMPLE_FROM_MODEL:
    model = load_model_unet(model_path, PARAMS["dropout_rate_model"], DEVICE)
    x_0_sample = torch.randn(PARAMS["amount_samples"], *x_0_train.shape[1:], device=DEVICE)
    solver = ODESolver(velocity_model=model)
    x_1_sample = solver.sample(x_init=x_0_sample, method=PARAMS["solver_method"], step_size=1.0 / PARAMS["solver_steps"])

    visualize_mnist_samples(x_1_sample, n_samples=PARAMS["amount_samples"])

