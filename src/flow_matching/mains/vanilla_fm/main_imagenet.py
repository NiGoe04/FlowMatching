import os

import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from torch.utils.data import DataLoader

from src.flow_matching.controller.cond_trainer import CondTrainer
from src.flow_matching.controller.lr_finder import LRFinder
from src.flow_matching.controller.utils import (
    store_model,
    get_imagenet_unet_model,
    load_model_unet_imagenet,
    load_imagenet_training_tensor,
)
from src.flow_matching.model.coupling import Coupler
from src.flow_matching.model.losses import ConditionalFMLoss
from src.flow_matching.view.utils import visualize_rgb_samples, visualize_rgb_trajectories

# steering console
DIM = 16  # supported by dataset files: 8, 16, 32, 64
FIND_LR = True
TRAIN_MODEL = True
SAVE_MODEL = True
GENERATE_SAMPLES = True
VISUALIZE_TIME = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "../../../../models"

# hyperparams
PARAMS = {
    "num_epochs": 8,
    "batch_size": 64,
    "dropout_rate_model": 0.0,
    "learning_rate": 2.5e-3,
    "size_train_set": 60000,
    "num_trainer_val_samples": 1000,
    "amount_samples": 8,
    "solver_steps": 100,
    "num_times_to_visualize": 9,
    "t_end": 1.05,
    "solver_method": "midpoint",
}

# data
x_1_train = load_imagenet_training_tensor(img_size=DIM, device=DEVICE, num_parts=1)
print(f"Training data shape: {x_1_train.shape}")
if PARAMS["size_train_set"] < x_1_train.shape[0]:
    x_1_train = x_1_train[: PARAMS["size_train_set"]]

x_0_train = torch.randn_like(x_1_train, device=DEVICE)

coupler = Coupler(x_0_train, x_1_train)
coupling = coupler.get_independent_coupling()
loader = DataLoader(coupling, PARAMS["batch_size"], shuffle=True)

# model
model = get_imagenet_unet_model(img_size=DIM, dropout_rate=PARAMS["dropout_rate_model"], device=DEVICE)

path = AffineProbPath(CondOTScheduler())
optimizer = torch.optim.Adam(model.parameters(), PARAMS["learning_rate"])
trainer = CondTrainer(model, optimizer, path, PARAMS["num_epochs"], PARAMS["num_trainer_val_samples"], device=DEVICE)

model_path = os.path.join(MODEL_SAVE_PATH, "model_IMAGENET_32_example.pth")


if FIND_LR:
    lr_finder = LRFinder(model, optimizer, path, ConditionalFMLoss(), device=DEVICE)
    lr_finder.range_test(loader, lr_start=1e-6, lr_end=1.0, num_iters=100)
    lr_finder.plot()

if TRAIN_MODEL:
    trainer.training_loop(loader)

if SAVE_MODEL:
    model_path = store_model(MODEL_SAVE_PATH, f"IMAGENET_{DIM}", model)

if GENERATE_SAMPLES:
    model = load_model_unet_imagenet(
        model_path=model_path,
        img_size=DIM,
        dropout_rate=PARAMS["dropout_rate_model"],
        device=DEVICE,
    )
    solver = ODESolver(velocity_model=model)
    x_0_sample = torch.randn(PARAMS["amount_samples"], 3, DIM, DIM, device=DEVICE)
    x_1_sample = solver.sample(
        x_init=x_0_sample,
        method=PARAMS["solver_method"],
        step_size=1.0 / PARAMS["solver_steps"],
        time_grid=torch.Tensor([0.0, PARAMS["t_end"]]).to(DEVICE),
    )
    visualize_rgb_samples(x_1_sample, 0, PARAMS["amount_samples"] - 1)

if VISUALIZE_TIME:
    model = load_model_unet_imagenet(
        model_path=model_path,
        img_size=DIM,
        dropout_rate=PARAMS["dropout_rate_model"],
        device=DEVICE,
    )
    solver = ODESolver(velocity_model=model)
    x_0_sample = torch.randn(PARAMS["amount_samples"], 3, DIM, DIM, device=DEVICE)
    time_grid = torch.linspace(0.0, PARAMS["t_end"], steps=PARAMS["num_times_to_visualize"], device=DEVICE)
    x_t = solver.sample(
        x_init=x_0_sample,
        method=PARAMS["solver_method"],
        step_size=1.0 / PARAMS["solver_steps"],
        return_intermediates=True,
        time_grid=time_grid,
    )
    sample_indices = list(range(min(3, PARAMS["amount_samples"])))
    visualize_rgb_trajectories(x_t, time_grid, sample_indices=sample_indices)
