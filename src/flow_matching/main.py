import os

import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from sklearn.datasets import make_moons
from torch import Tensor
from torch.utils.data import DataLoader

from src.flow_matching.controller.cond_trainer import CondTrainer
from src.flow_matching.controller.utils import plot_tensor_2d, store_model, load_model
from src.flow_matching.model.coupling import Coupler
from src.flow_matching.model.velocity_field import SimpleVelocityModel

# steering console
TRAIN_MODEL = False
SAVE_MODEL = False
SAMPLE_FROM_MODEL = True
CREATE_SAMPLES = True

MODEL_SAVE_PATH = "../../models"
# hyperparams
PARAMS = {
    "num_epochs": 10,
    "batch_size": 8,
    "learning_rate": 1e-2,
    "size_train_set": 100000,
    "amount_samples": 1000,
    "solver_steps": 100,
}

# data
x_0 = torch.randn(PARAMS["size_train_set"], 2)
x_1 = Tensor(make_moons(PARAMS["size_train_set"], noise=0.05)[0])
coupler = Coupler(x_0, x_1)
coupling = coupler.get_independent_coupling()
loader = DataLoader(
    coupling,
    PARAMS["batch_size"],
    shuffle=True,
)

# model and training
model = SimpleVelocityModel()
path = AffineProbPath(CondOTScheduler())
optimizer = torch.optim.Adam(model.parameters(), PARAMS["learning_rate"])
trainer = CondTrainer(model, optimizer, path, PARAMS["num_epochs"])
model_path = os.path.join(MODEL_SAVE_PATH, "model_2025-11-02_23-01-38.pth")
if TRAIN_MODEL:
    trainer.training_loop(loader)
if SAVE_MODEL:
    # noinspection PyRedeclaration
    model_path = store_model(MODEL_SAVE_PATH, model)

if SAMPLE_FROM_MODEL:
    model = load_model(SimpleVelocityModel, model_path)
    x_0 = torch.randn(PARAMS["amount_samples"], 2)
    solver = ODESolver(velocity_model=model)
    x_1 = solver.sample(x_init=x_0, method='midpoint', step_size=1.0 / PARAMS["solver_steps"])

    plot_tensor_2d(x_1)

