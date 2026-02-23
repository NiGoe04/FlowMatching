from __future__ import annotations

import os
from typing import Optional

import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from torch.utils.data import DataLoader

from src.flow_matching.controller.cond_trainer import CondTrainerBatchOT, CondTrainer
from src.flow_matching.controller.utils import store_model
from src.flow_matching.model.coupling import Coupler
from src.flow_matching.model.velocity_model_basic import SimpleVelocityModel

REGISTRY_FILE = os.path.join(os.path.dirname(__file__), "current_model_paths.py")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")


def _solver_display_name(ot_optimizer: str, epsilon: Optional[float]) -> str:
    if ot_optimizer == "hungarian":
        return "hungarian"
    if ot_optimizer == "sinkhorn":
        if epsilon is None:
            raise ValueError("Sinkhorn optimizer requires an epsilon value.")
        return f"sinkhorn_{epsilon}"
    raise ValueError(f"Unsupported ot_optimizer: {ot_optimizer}")


def build_registry_key(dim: int, ot_batch_size: int, ot_optimizer: str, scenario_name: str, epsilon: Optional[float], vanilla_fm_mode) -> str:
    if ot_optimizer == "hungarian":
        if vanilla_fm_mode:
            return f"{dim}D|{ot_batch_size}V|{ot_optimizer}|{scenario_name}"
        else:
            return f"{dim}D|{ot_batch_size}N|{ot_optimizer}|{scenario_name}"
    if vanilla_fm_mode:
        return f"{dim}D|{ot_batch_size}V|{ot_optimizer}|eps={epsilon}|{scenario_name}"
    else:
        return f"{dim}D|{ot_batch_size}N|{ot_optimizer}|eps={epsilon}|{scenario_name}"


def _persist_registry(model_paths: dict[str, str]) -> None:
    lines = [
        '"""Auto-managed registry of trained model paths for mass experiments."""',
        "",
        "# key format:",
        "# \"{dim}D|{ot_batch_size}V|{ot_solver_name}|{scenario_name}\" for hungarian, vanilla fm mode",
        "# \"{dim}D|{ot_batch_size}V|{ot_solver_name}|eps={epsilon}|{scenario_name}\" for sinkhorn, vanilla fm mode",
        "# \"{dim}D|{ot_batch_size}N|{ot_solver_name}|{scenario_name}\" for hungarian, batch OT mode",
        "# \"{dim}D|{ot_batch_size}N|{ot_solver_name}|eps={epsilon}|{scenario_name}\" for sinkhorn, batch OT mode",
        "MODEL_PATHS: dict[str, str] = {",
    ]
    for key in sorted(model_paths):
        lines.append(f"    {key!r}: {model_paths[key]!r},")
    lines.append("}")
    lines.append("")

    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _load_registry() -> dict[str, str]:
    namespace: dict[str, object] = {}
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        exec(content, namespace)
    value = namespace.get("MODEL_PATHS", {})
    if not isinstance(value, dict):
        raise ValueError("MODEL_PATHS in registry must be a dict.")
    return {str(k): str(v) for k, v in value.items()}


def train_or_get_model(
    *,
    dim: int,
    scenario_name: str,
    gmd_x0,
    gmd_x1,
    ot_batch_size: int,
    ot_optimizer: str,
    epsilon: Optional[float],
    params_exp: dict,
    vanilla_fm_mode,
    device: torch.device,
) -> str:
    """
    Trains a model once per unique key and writes/reads src/experiments/current_model_paths.py.
    Returns the path to the trained model.
    """
    if ot_optimizer != "hungarian":
        raise ValueError(
            f"OT optimizer '{ot_optimizer}' is not implemented. Use 'hungarian'."
        )

    registry = _load_registry()
    key = build_registry_key(dim, ot_batch_size, ot_optimizer, scenario_name, epsilon, vanilla_fm_mode)
    existing = registry.get(key)
    if existing and os.path.exists(existing):
        print(f"Using existing registered model. Vanilla FM mode: {vanilla_fm_mode}")
        return existing

    x0_train = gmd_x0.sample(int(params_exp["size_train_set"]))
    x1_train = gmd_x1.sample(int(params_exp["size_train_set"]))

    coupler = Coupler(x0_train, x1_train)
    coupling = coupler.get_independent_coupling()

    train_loader = DataLoader(
        coupling,
        ot_batch_size,
        shuffle=True,
    )

    model = SimpleVelocityModel(device=device, dim=dim)
    optimizer = torch.optim.Adam(model.parameters(), float(params_exp["learning_rate"]))
    path = AffineProbPath(CondOTScheduler())
    if vanilla_fm_mode:
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
        )

    trainer.training_loop(train_loader)

    solver_name = _solver_display_name(ot_optimizer, epsilon)
    if vanilla_fm_mode:
        model_name = f"{dim}D_{ot_batch_size}V_{solver_name}_{scenario_name}"
    else:
        model_name = f"{dim}D_{ot_batch_size}N_{solver_name}_{scenario_name}"
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = store_model(MODELS_DIR, model_name, model)

    registry[key] = model_path
    _persist_registry(registry)
    return model_path
