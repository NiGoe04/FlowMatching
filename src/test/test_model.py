from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from flow_matching.solver import ODESolver

from src.experiments.mixed_gaussian_framework.current_model_paths import MODEL_PATHS
from src.experiments.mixed_gaussian_framework.mass_training_mixed_gaussian import build_registry_key
from src.experiments.mixed_gaussian_framework.scenarios import SCENARIO_NAMES, get_scenario
from src.flow_matching.controller.utils import load_model_n_dim


# =========================
# 🔧 CONFIGURATION (EDIT HERE)
# =========================
DIM = 8
OT_BATCH_SIZE = 1
SCENARIO = "tri_gauss_twice"  # or set explicitly as string
OT_OPTIMIZER = "hungarian"  # "hungarian" or "sinkhorn"
EPSILON: Optional[float] = None  # required if sinkhorn
VANILLA_FM_MODE = False

AMOUNT_SAMPLES = 16
SOLVER_METHOD = "midpoint"
SOLVER_STEPS = 100
T_END = 1.0
SEED = 42
# =========================


def _solver_display_name(ot_optimizer: str, epsilon: Optional[float]) -> str:
    if ot_optimizer == "hungarian":
        return "hungarian"
    if ot_optimizer == "sinkhorn":
        if epsilon is None:
            raise ValueError("Sinkhorn optimizer requires epsilon.")
        return f"sinkhorn_{epsilon}"
    raise ValueError(f"Unsupported ot optimizer: {ot_optimizer}")


def _resolve_model_path(
    *,
    framework_models_dir: Path,
    dim: int,
    ot_batch_size: int,
    ot_optimizer: str,
    epsilon: Optional[float],
    scenario_name: str,
    vanilla_fm_mode: bool,
) -> Path:
    key = build_registry_key(dim, ot_batch_size, ot_optimizer, scenario_name, epsilon, vanilla_fm_mode)
    candidate = MODEL_PATHS.get(key)

    if candidate is not None:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

        basename_fallback = framework_models_dir / candidate_path.name
        if basename_fallback.exists():
            return basename_fallback

    solver_name = _solver_display_name(ot_optimizer, epsilon)
    mode_char = "V" if vanilla_fm_mode else "N"
    pattern = f"model_{dim}D_{ot_batch_size}{mode_char}_{solver_name}_{scenario_name}_*.pth"
    matches = sorted(framework_models_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"No matching model found for key '{key}' and pattern '{pattern}' in '{framework_models_dir}'."
    )


def main() -> None:
    if OT_OPTIMIZER == "sinkhorn" and EPSILON is None:
        raise ValueError("EPSILON must be set when using sinkhorn.")

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    framework_dir = Path(__file__).resolve().parents[1] / "experiments" / "mixed_gaussian_framework"
    models_dir = framework_dir / "models"

    model_path = _resolve_model_path(
        framework_models_dir=models_dir,
        dim=DIM,
        ot_batch_size=OT_BATCH_SIZE,
        ot_optimizer=OT_OPTIMIZER,
        epsilon=EPSILON,
        scenario_name=SCENARIO,
        vanilla_fm_mode=VANILLA_FM_MODE,
    )

    gmd_x0, gmd_x1, _ = get_scenario(scenario_name=SCENARIO, dim=DIM, device=device)
    x0_samples = gmd_x0.sample(AMOUNT_SAMPLES)

    model = load_model_n_dim(DIM, str(model_path), device=device)
    solver = ODESolver(velocity_model=model)

    generated_target_samples = solver.sample(
        x_init=x0_samples,
        method=SOLVER_METHOD,
        step_size=1.0 / float(SOLVER_STEPS),
        time_grid=torch.tensor([0.0, float(T_END)], device=device),
    )

    true_target_samples = gmd_x1.sample(AMOUNT_SAMPLES)

    print(f"Using device: {device}")
    print(f"Using model: {model_path}")
    print(f"Scenario: {SCENARIO} | dim={DIM} | ot_batch_size={OT_BATCH_SIZE}")

    print("\nSource samples x0:")
    print(x0_samples)

    print("\nGenerated target samples (model output):")
    print(generated_target_samples)

    print("\nReference target samples (fresh draw from target distribution x1):")
    print(true_target_samples)


if __name__ == "__main__":
    main()