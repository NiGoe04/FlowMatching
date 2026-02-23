from __future__ import annotations

import itertools
import os

import torch

from src.experiments.create_report import create_experiment_report
from src.experiments.mass_training import train_or_get_model
from src.experiments.scenarios import SCENARIO_NAMES, get_scenario
from src.flow_matching.controller.metrics import Metrics
from src.flow_matching.controller.utils import load_model_n_dim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DO_MASS_TRAINING = True

SCENARIOS = ["double_gauss_twice", "double_gauss_twice_ftd"]
DIMS = [2, 3, 5, 8, 21, 89, 144, 233, 377, 610]
OT_BATCH_SIZES = [64, 128, 256]
OT_OPTIMIZERS = ["hungarian"]

# Optional epsilon values are only used if a sinkhorn optimizer is configured.
SINKHORN_EPSILONS = [0.01]

PARAMS_EXP = {
    "num_epochs": 8,
    "learning_rate": 1.8e-3,
    "size_train_set": 200000,
    "amount_samples": 1000,
    "num_trainer_val_samples": 5000,
    "batch_size": 256,
}


def _optimizer_and_epsilon_grid() -> list[tuple[str, float | None]]:
    out: list[tuple[str, float | None]] = []
    for optimizer_name in OT_OPTIMIZERS:
        if optimizer_name == "hungarian":
            out.append((optimizer_name, None))
            continue
        raise ValueError(
            "Sinkhorn is not implemented. Please only use 'hungarian' in OT_OPTIMIZERS. "
            f"Found: {optimizer_name}"
        )
    return out


def _compute_metrics(model, gmd_x0, gmd_x1, w2_sq_pre_calc: float | None, amount_samples: int) -> dict[str, float]:
    x0_eval = gmd_x0.sample(amount_samples)
    x1_eval = gmd_x1.sample(amount_samples)

    straightness = Metrics.calculate_path_straightness(model, x0_eval)
    pe, w2_sq, npe = Metrics.calculate_normalized_path_energy(model, x0_eval, x1_eval, w2_sq_pre_calc)
    _, psi_1 = Metrics._calculate_mean_velocity_norm_sq(model, x0_eval)

    nll = gmd_x1.nll(psi_1)
    nll_mi_corr = gmd_x1.nll_mi_corrected(psi_1)
    nll_per_dim = gmd_x1.nll_per_dim(psi_1)
    nll_mi_corr_per_dim = gmd_x1.nll_mi_corrected_per_dim(psi_1)

    return {
        "straightness": float(straightness.item()),
        "path_energy": float(pe.item()),
        "w2_sq": float(w2_sq.item()),
        "normalized_path_energy": float(npe.item()),
        "nll": float(nll.item()),
        "nll_mi_corrected": float(nll_mi_corr.item()),
        "nll_per_dim": float(nll_per_dim.item()),
        "nll_mi_corrected_per_dim": float(nll_mi_corr_per_dim.item()),
    }


def run() -> str:
    unknown_scenarios = [name for name in SCENARIOS if name not in SCENARIO_NAMES]
    if unknown_scenarios:
        raise ValueError(f"Unknown scenarios configured: {unknown_scenarios}. Available: {SCENARIO_NAMES}")

    all_results: list[dict] = []

    grid = itertools.product(
        SCENARIOS,
        DIMS,
        OT_BATCH_SIZES,
        _optimizer_and_epsilon_grid(),
    )

    for scenario_name, dim, ot_batch_size, (ot_optimizer, epsilon) in grid:
        gmd_x0, gmd_x1, _, w2_sq_pre_calc = get_scenario(scenario_name, dim, DEVICE)

        if DO_MASS_TRAINING:
            model_path = train_or_get_model(
                dim=dim,
                scenario_name=scenario_name,
                gmd_x0=gmd_x0,
                gmd_x1=gmd_x1,
                ot_batch_size=ot_batch_size,
                ot_optimizer=ot_optimizer,
                epsilon=epsilon,
                params_exp=PARAMS_EXP,
                device=DEVICE,
            )
        else:
            raise RuntimeError(
                "DO_MASS_TRAINING is disabled, but automated fetch-only mode is not configured. "
                "Enable DO_MASS_TRAINING or add a custom model-loading workflow."
            )

        model = load_model_n_dim(dim, model_path, device=DEVICE)
        metrics = _compute_metrics(
            model=model,
            gmd_x0=gmd_x0,
            gmd_x1=gmd_x1,
            w2_sq_pre_calc=w2_sq_pre_calc,
            amount_samples=int(PARAMS_EXP["amount_samples"]),
        )

        all_results.append(
            {
                "combination": {
                    "scenario": scenario_name,
                    "dim": dim,
                    "ot_batch_size": ot_batch_size,
                    "ot_optimizer": ot_optimizer,
                    "epsilon": epsilon,
                    "model_path": model_path,
                },
                "metrics": metrics,
            }
        )

    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    return create_experiment_report(all_results, reports_dir)


if __name__ == "__main__":
    report = run()
    print(f"Saved report to: {report}")
