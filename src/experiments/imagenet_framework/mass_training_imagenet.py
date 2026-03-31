from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from torch.utils.data import DataLoader

from src.experiments.imagenet_framework.utils import REGISTRY_FILE, MODELS_DIR
from src.flow_matching.controller.cond_trainer import CondTrainer, CondTrainerBatchOT
from src.flow_matching.controller.utils import store_model, get_imagenet_unet_model
from src.flow_matching.model.coupling import Coupler

STANDARD_OPTIMIZATION_BATCH_SIZE = 256


def build_registry_key(dim: int, ot_batch_size: int, ot_optimizer: str, epsilon: Optional[float]) -> str:
    if ot_optimizer == "hungarian":
        return f"imagenet|{dim}|k={ot_batch_size}|{ot_optimizer}"
    return f"imagenet|{dim}|k={ot_batch_size}|{ot_optimizer}|eps={epsilon}"


def _persist_registry(model_paths: dict[str, str]) -> None:
    lines = [
        '"""Auto-managed registry of trained model paths for imagenet framework."""',
        "",
        "MODEL_PATHS: dict[str, str] = {",
    ]
    for key in sorted(model_paths):
        lines.append(f"    {key!r}: {model_paths[key]!r},")
    lines.append("}")
    lines.append("")
    REGISTRY_FILE.write_text("\n".join(lines), encoding="utf-8")


def _load_registry() -> dict[str, str]:
    namespace: dict[str, object] = {}
    if os.path.exists(REGISTRY_FILE):
        content = REGISTRY_FILE.read_text(encoding="utf-8")
        exec(content, namespace)
    value = namespace.get("MODEL_PATHS", {})
    if not isinstance(value, dict):
        raise ValueError("MODEL_PATHS in registry must be a dict.")
    return {str(k): str(v) for k, v in value.items()}


def train_or_get_model(
    *,
    dim: int,
    x0_train: torch.Tensor,
    x1_train: torch.Tensor,
    ot_batch_size: int,
    ot_optimizer: str,
    epsilon: Optional[float],
    params_exp: dict,
    device: torch.device,
) -> tuple[str, list[float], list[float], bool]:
    registry = _load_registry()
    key = build_registry_key(dim, ot_batch_size, ot_optimizer, epsilon)
    existing = registry.get(key)
    if existing and os.path.exists(existing):
        print("Using existing registered model.")
        return existing, [], [], False

    coupling = Coupler(x0_train, x1_train).get_independent_coupling()
    batch_size = STANDARD_OPTIMIZATION_BATCH_SIZE if ot_batch_size == 1 else ot_batch_size
    train_loader = DataLoader(coupling, batch_size, shuffle=True)

    model = get_imagenet_unet_model(
        img_size=dim,
        dropout_rate=float(params_exp["dropout_rate_model"]),
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), float(params_exp["learning_rate"]))
    path = AffineProbPath(CondOTScheduler())

    if ot_batch_size == 1:
        trainer = CondTrainer(
            model=model,
            optimizer=optimizer,
            path=path,
            num_epochs=int(params_exp["num_epochs"]),
            num_val_samples=int(params_exp["num_trainer_val_samples"]),
            device=device,
        )
    else:
        trainer = CondTrainerBatchOT(
            model=model,
            optimizer=optimizer,
            path=path,
            num_epochs=int(params_exp["num_epochs"]),
            num_val_samples=int(params_exp["num_trainer_val_samples"]),
            device=device,
            use_sinkhorn=(ot_optimizer == "sinkhorn"),
            sinkhorn_eps=(0.1 if epsilon is None else float(epsilon)),
        )

    train_losses, val_losses = trainer.training_loop(train_loader)

    os.makedirs(MODELS_DIR, exist_ok=True)
    solver_name = ot_optimizer if ot_optimizer == "hungarian" else f"sinkhorn_{epsilon}"
    model_path = store_model(str(MODELS_DIR), f"imagenet_{dim}_k{ot_batch_size}_{solver_name}", model)

    registry[key] = model_path
    _persist_registry(registry)
    return model_path, train_losses, val_losses, True
