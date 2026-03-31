from __future__ import annotations

from pathlib import Path

import torch
from flow_matching.solver import ODESolver

from src.flow_matching.controller.utils import load_imagenet_training_tensor

FRAMEWORK_DIR = Path(__file__).resolve().parent
REGISTRY_FILE = FRAMEWORK_DIR / "current_model_paths.py"
MODELS_DIR = FRAMEWORK_DIR / "models"
LOSS_PLOTS_DIR = FRAMEWORK_DIR / "loss_plots"
METRICS_PLOTS_DIR = FRAMEWORK_DIR / "metrics_plots"
TABLES_DIR = FRAMEWORK_DIR / "tables"
REPORTS_DIR = FRAMEWORK_DIR / "reports"

def load_imagenet_scenario_data(dim: int, size_train_set: int, device: torch.device) -> torch.Tensor:
    x = load_imagenet_training_tensor(img_size=dim, device=device)
    if size_train_set < x.shape[0]:
        x = x[:size_train_set]
    return x


def sample_model_images(model, dim: int, amount_samples: int, solver_steps: int, t_end: float, device: torch.device) -> torch.Tensor:
    solver = ODESolver(velocity_model=model)
    x0 = torch.randn(amount_samples, 3, dim, dim, device=device)
    x1 = solver.sample(
        x_init=x0,
        method="midpoint",
        step_size=1.0 / solver_steps,
        time_grid=torch.tensor([0.0, t_end], device=device),
    )
    return x1
