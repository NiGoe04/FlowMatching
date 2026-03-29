from __future__ import annotations

import itertools
import os
from pathlib import Path

import torch

from src.experiments.mixed_gaussian_framework.create_report import create_experiment_report
from src.experiments.mixed_gaussian_framework.scenarios import SCENARIO_NAMES, get_scenario
from src.experiments.mixed_gaussian_framework.mass_training_mixed_gaussian import (
    train_or_get_model,
    save_loss_plot,
)
from src.flow_matching.controller.metrics import Metrics
from src.flow_matching.controller.utils import load_model_n_dim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_VANILLA_FOR_COMPARISON = True
SAVE_LOSS_PLOTS = True

SCENARIOS = ["4_to_4_gauss"]
DIMS = [2, 1024]
OT_BATCH_SIZES = [256]
OT_OPTIMIZERS = ["hungarian"]

# Optional epsilon values are only used if a sinkhorn optimizer is configured.
SINKHORN_EPSILONS = [0.01, 0.1]

PARAMS_EXP = {
    "num_epochs": 10,
    "learning_rate": 1.8e-3,
    "size_train_set": 80000,
    "amount_samples": 5000,
    "num_trainer_val_samples": 5000,
}


def _optimizer_and_epsilon_grid() -> list[tuple[str, float | None]]:
    out: list[tuple[str, float | None]] = []
    for optimizer_name in OT_OPTIMIZERS:
        if optimizer_name == "hungarian":
            out.append((optimizer_name, None))
        elif optimizer_name == "sinkhorn":
            for epsilon in SINKHORN_EPSILONS:
                out.append((optimizer_name, float(epsilon)))
        else:
            raise ValueError(
                f"Unsupported optimizer in OT_OPTIMIZERS: {optimizer_name}. "
                "Supported values: ['hungarian', 'sinkhorn']."
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


def _maybe_save_loss_plot(model_path: str, train_losses: list[float], val_losses: list[float], model_was_trained: bool):
    if not SAVE_LOSS_PLOTS or not model_was_trained:
        return None
    if not train_losses or not val_losses:
        return None
    return save_loss_plot(model_path=model_path, train_losses=train_losses, val_losses=val_losses)


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
        gmd_x0, gmd_x1, w2_sq_pre_calc = get_scenario(scenario_name, dim, DEVICE)

        if TRAIN_VANILLA_FOR_COMPARISON:
            print(
                f"[SCENARIO] name={scenario_name} | "
                f"dim={dim} | "
                f"batch_size={ot_batch_size} | "
                f"optimizer={ot_optimizer} | "
                f"epsilon={epsilon} | "
                f"Vanilla FM"
            )

            model_path_vanilla, train_losses, val_losses, trained = train_or_get_model(
                dim=dim,
                scenario_name=scenario_name,
                gmd_x0=gmd_x0,
                gmd_x1=gmd_x1,
                ot_batch_size=ot_batch_size,
                ot_optimizer=ot_optimizer,
                epsilon=epsilon,
                params_exp=PARAMS_EXP,
                vanilla_fm_mode=True,
                device=DEVICE,
            )
            loss_plot_path = _maybe_save_loss_plot(model_path_vanilla, train_losses, val_losses, trained)

            model_vanilla = load_model_n_dim(dim, model_path_vanilla, device=DEVICE)
            metrics_vanilla = _compute_metrics(
                model=model_vanilla,
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
                        "model_path": model_path_vanilla,
                        "loss_plot_path": loss_plot_path,
                        "vanilla": True,
                    },
                    "metrics": metrics_vanilla,
                }
            )

        print(
            f"[SCENARIO] name={scenario_name} | "
            f"dim={dim} | "
            f"batch_size={ot_batch_size} | "
            f"optimizer={ot_optimizer} | "
            f"epsilon={epsilon} | "
            f"OT-CFM"
        )

        model_path_ot_cfm, train_losses, val_losses, trained = train_or_get_model(
            dim=dim,
            scenario_name=scenario_name,
            gmd_x0=gmd_x0,
            gmd_x1=gmd_x1,
            ot_batch_size=ot_batch_size,
            ot_optimizer=ot_optimizer,
            epsilon=epsilon,
            params_exp=PARAMS_EXP,
            vanilla_fm_mode=False,
            device=DEVICE,
        )
        loss_plot_path = _maybe_save_loss_plot(model_path_ot_cfm, train_losses, val_losses, trained)

        model_ot_cfm = load_model_n_dim(dim, model_path_ot_cfm, device=DEVICE)
        metrics_ot_cfm = _compute_metrics(
            model=model_ot_cfm,
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
                    "model_path": model_path_ot_cfm,
                    "loss_plot_path": loss_plot_path,
                    "vanilla": False,
                },
                "metrics": metrics_ot_cfm,
            }
        )

    reports_dir = os.path.join(Path(__file__).resolve().parent, "reports")
    return create_experiment_report(all_results, reports_dir)


if __name__ == "__main__":
    report = run()
    print(f"Saved report to: {report}")
