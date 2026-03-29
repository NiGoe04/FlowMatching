from __future__ import annotations

import itertools
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.experiments.mixed_gaussian_framework.create_report import create_experiment_report
from src.experiments.mixed_gaussian_framework.mass_training_mixed_gaussian import train_or_get_model
from src.experiments.mixed_gaussian_framework.scenarios import SCENARIO_NAMES, get_scenario
from src.flow_matching.controller.metrics import Metrics
from src.flow_matching.controller.utils import load_model_n_dim
from src.view.utils import ensure_dir, make_timestamp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_VANILLA_FOR_COMPARISON = True
SAVE_LOSS_PLOTS = True
SAVE_METRICS_PLOTS = True
ITERATIONS = 6

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

FRAMEWORK_DIR = Path(__file__).resolve().parent
METRICS_PLOTS_DIR = FRAMEWORK_DIR / "metrics_plots"
TABLES_DIR = FRAMEWORK_DIR / "tables"
LOSS_PLOTS_DIR = FRAMEWORK_DIR / "loss_plots"

METRIC_KEYS = [
    "path_energy",
    "normalized_path_energy",
    "straightness",
    "nll_per_dim",
    "nll_mi_corrected_per_dim",
]

METRIC_LATEX_LABELS = {
    "path_energy": r"$\mathrm{PE}(\theta)$",
    "normalized_path_energy": r"$\mathrm{NPE}(\theta)$",
    "straightness": r"$S(\theta)$",
    "nll_per_dim": r"$\mathrm{NLL}(\theta)/d$",
    "nll_mi_corrected_per_dim": r"$\mathrm{NLL}_{\mathrm{corr}}(\theta)/d$",
}

METRIC_FILENAMES = {
    "path_energy": "path_energy",
    "normalized_path_energy": "normalized_path_energy",
    "straightness": "straightness",
    "nll_per_dim": "nll_per_dim",
    "nll_mi_corrected_per_dim": "nll_mi_corrected_per_dim",
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


def _mean_and_unbiased_std(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    mean_val = float(tensor.mean().item())
    if tensor.numel() < 2:
        return mean_val, 0.0
    std_val = float(tensor.std(unbiased=True).item())
    return mean_val, std_val


def _sanitize_filename_part(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _save_loss_plots_by_dimension(
    loss_by_group: dict[tuple[str, int], dict[str, dict[int, list[float]]]],
    timestamp: str,
) -> list[str]:
    ensure_dir(LOSS_PLOTS_DIR)
    saved_paths: list[str] = []

    for (scenario_name, ot_batch_size), losses_by_kind in sorted(loss_by_group.items()):
        for loss_kind, losses_by_dim in losses_by_kind.items():
            if not losses_by_dim:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            for dim in sorted(losses_by_dim):
                values = losses_by_dim[dim]
                epochs = range(1, len(values) + 1)
                ax.plot(epochs, values, linewidth=2, label=f"d={dim}")

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            title_kind = "training" if loss_kind == "train" else "validation"
            ax.set_title(f"{scenario_name} | k={ot_batch_size} | {title_kind} loss")
            ax.legend()
            ax.grid(alpha=0.2)

            filename = (
                f"scenario_{_sanitize_filename_part(scenario_name)}"
                f"_k_{ot_batch_size}_{loss_kind}_loss_{timestamp}.png"
            )
            out_path = LOSS_PLOTS_DIR / filename
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            saved_paths.append(str(out_path))

    return saved_paths


def _save_metrics_plots(
    metrics_by_group: dict[tuple[str, int, int], dict[str, list[float]]],
    timestamp: str,
) -> list[str]:
    ensure_dir(METRICS_PLOTS_DIR)
    saved_paths: list[str] = []

    scenarios = sorted({key[0] for key in metrics_by_group})

    for scenario_name in scenarios:
        dims = sorted({key[1] for key in metrics_by_group if key[0] == scenario_name})
        ot_batch_sizes = sorted({key[2] for key in metrics_by_group if key[0] == scenario_name})

        for metric_key in METRIC_KEYS:
            fig, ax = plt.subplots(figsize=(8, 5))

            for ot_batch_size in ot_batch_sizes:
                mean_values = []
                for dim in dims:
                    values = metrics_by_group[(scenario_name, dim, ot_batch_size)][metric_key]
                    mean_val, _ = _mean_and_unbiased_std(values)
                    mean_values.append(mean_val)

                ax.plot(dims, mean_values, marker="o", linewidth=2, label=f"k = {ot_batch_size}")

            ax.set_xlabel("d")
            ax.set_ylabel(METRIC_LATEX_LABELS[metric_key])
            ax.grid(alpha=0.2)
            ax.legend()

            filename = (
                f"scenario_{_sanitize_filename_part(scenario_name)}"
                f"_metric_{METRIC_FILENAMES[metric_key]}_{timestamp}.png"
            )
            out_path = METRICS_PLOTS_DIR / filename
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            saved_paths.append(str(out_path))

    return saved_paths


def _build_metrics_table_latex(
    *,
    scenario_name: str,
    ot_batch_size: int,
    dims: list[int],
    metrics_by_group: dict[tuple[str, int, int], dict[str, list[float]]],
) -> str:
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "",
        "\\begin{tabular}{c|c c c c c}",
        "\\toprule",
        "$d$ & $\\mathrm{PE}(\\theta)$ & $\\mathrm{NPE}(\\theta)$ & $S(\\theta)$ & $\\mathrm{NLL}(\\theta)/d$ & $\\mathrm{NLL}_{\\mathrm{corr}}(\\theta)/d$ " + r"\\",
        "\\midrule",
    ]

    for dim in dims:
        stats = {}
        for metric_key in METRIC_KEYS:
            mean_val, std_val = _mean_and_unbiased_std(metrics_by_group[(scenario_name, dim, ot_batch_size)][metric_key])
            stats[metric_key] = (mean_val, std_val)

        row = (
            f"{dim}"
            f" & \\num{{{stats['path_energy'][0]:.6g} +- {stats['path_energy'][1]:.3g}}}"
            f" & \\num{{{stats['normalized_path_energy'][0]:.6g} +- {stats['normalized_path_energy'][1]:.3g}}}"
            f" & \\num{{{stats['straightness'][0]:.6g} +- {stats['straightness'][1]:.3g}}}"
            f" & \\num{{{stats['nll_per_dim'][0]:.6g} +- {stats['nll_per_dim'][1]:.3g}}}"
            f" & \\num{{{stats['nll_mi_corrected_per_dim'][0]:.6g} +- {stats['nll_mi_corrected_per_dim'][1]:.3g}}} "
            + r"\\"
        )
        lines.append(row)

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            f"\\caption{{Performance metrics across dimensions for scenario '{scenario_name}' and OT batch size $k={ot_batch_size}$.}}",
            f"\\label{{tab:metrics_{_sanitize_filename_part(scenario_name)}_k_{ot_batch_size}}}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def _save_metrics_tables(
    metrics_by_group: dict[tuple[str, int, int], dict[str, list[float]]],
    timestamp: str,
) -> list[str]:
    ensure_dir(TABLES_DIR)
    saved_paths: list[str] = []

    scenarios = sorted({key[0] for key in metrics_by_group})

    for scenario_name in scenarios:
        dims = sorted({key[1] for key in metrics_by_group if key[0] == scenario_name})
        ot_batch_sizes = sorted({key[2] for key in metrics_by_group if key[0] == scenario_name})

        for ot_batch_size in ot_batch_sizes:
            latex_content = _build_metrics_table_latex(
                scenario_name=scenario_name,
                ot_batch_size=ot_batch_size,
                dims=dims,
                metrics_by_group=metrics_by_group,
            )
            filename = (
                f"scenario_{_sanitize_filename_part(scenario_name)}"
                f"_k_{ot_batch_size}_{timestamp}.tex"
            )
            out_path = TABLES_DIR / filename
            out_path.write_text(latex_content, encoding="utf-8")
            saved_paths.append(str(out_path))

    return saved_paths


def _compute_metrics_for_iterations(
    *,
    model,
    gmd_x0,
    gmd_x1,
    w2_sq_pre_calc: float | None,
    amount_samples: int,
) -> dict[str, list[float]]:
    metrics_per_iteration: dict[str, list[float]] = defaultdict(list)

    for iteration_idx in range(ITERATIONS):
        iter_metrics = _compute_metrics(
            model=model,
            gmd_x0=gmd_x0,
            gmd_x1=gmd_x1,
            w2_sq_pre_calc=w2_sq_pre_calc,
            amount_samples=amount_samples,
        )
        for metric_name, metric_value in iter_metrics.items():
            metrics_per_iteration[metric_name].append(metric_value)
        print(f"  Metrics iteration {iteration_idx + 1}/{ITERATIONS} done.")

    return dict(metrics_per_iteration)


def run() -> str:
    unknown_scenarios = [name for name in SCENARIOS if name not in SCENARIO_NAMES]
    if unknown_scenarios:
        raise ValueError(f"Unknown scenarios configured: {unknown_scenarios}. Available: {SCENARIO_NAMES}")

    timestamp = make_timestamp()

    all_results: list[dict] = []
    metrics_by_group: dict[tuple[str, int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    loss_by_group: dict[tuple[str, int], dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: {"train": {}, "val": {}}
    )

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

            if SAVE_LOSS_PLOTS and trained and train_losses and val_losses:
                loss_by_group[(scenario_name, ot_batch_size)]["train"][dim] = train_losses
                loss_by_group[(scenario_name, ot_batch_size)]["val"][dim] = val_losses

            model_vanilla = load_model_n_dim(dim, model_path_vanilla, device=DEVICE)
            metrics_vanilla_all = _compute_metrics_for_iterations(
                model=model_vanilla,
                gmd_x0=gmd_x0,
                gmd_x1=gmd_x1,
                w2_sq_pre_calc=w2_sq_pre_calc,
                amount_samples=int(PARAMS_EXP["amount_samples"]),
            )

            metrics_vanilla_mean = {
                metric_name: _mean_and_unbiased_std(metric_values)[0]
                for metric_name, metric_values in metrics_vanilla_all.items()
            }
            metrics_vanilla_std = {
                f"{metric_name}_std": _mean_and_unbiased_std(metric_values)[1]
                for metric_name, metric_values in metrics_vanilla_all.items()
            }

            all_results.append(
                {
                    "combination": {
                        "scenario": scenario_name,
                        "dim": dim,
                        "ot_batch_size": ot_batch_size,
                        "ot_optimizer": ot_optimizer,
                        "epsilon": epsilon,
                        "model_path": model_path_vanilla,
                        "vanilla": True,
                    },
                    "metrics": {**metrics_vanilla_mean, **metrics_vanilla_std},
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

        if SAVE_LOSS_PLOTS and trained and train_losses and val_losses:
            loss_by_group[(scenario_name, ot_batch_size)]["train"][dim] = train_losses
            loss_by_group[(scenario_name, ot_batch_size)]["val"][dim] = val_losses

        model_ot_cfm = load_model_n_dim(dim, model_path_ot_cfm, device=DEVICE)
        metrics_ot_cfm_all = _compute_metrics_for_iterations(
            model=model_ot_cfm,
            gmd_x0=gmd_x0,
            gmd_x1=gmd_x1,
            w2_sq_pre_calc=w2_sq_pre_calc,
            amount_samples=int(PARAMS_EXP["amount_samples"]),
        )

        metrics_ot_cfm_mean = {
            metric_name: _mean_and_unbiased_std(metric_values)[0]
            for metric_name, metric_values in metrics_ot_cfm_all.items()
        }
        metrics_ot_cfm_std = {
            f"{metric_name}_std": _mean_and_unbiased_std(metric_values)[1]
            for metric_name, metric_values in metrics_ot_cfm_all.items()
        }

        for metric_name in METRIC_KEYS:
            metrics_by_group[(scenario_name, dim, ot_batch_size)][metric_name].extend(metrics_ot_cfm_all[metric_name])

        all_results.append(
            {
                "combination": {
                    "scenario": scenario_name,
                    "dim": dim,
                    "ot_batch_size": ot_batch_size,
                    "ot_optimizer": ot_optimizer,
                    "epsilon": epsilon,
                    "model_path": model_path_ot_cfm,
                    "vanilla": False,
                },
                "metrics": {**metrics_ot_cfm_mean, **metrics_ot_cfm_std},
            }
        )

    if SAVE_LOSS_PLOTS:
        saved_loss_paths = _save_loss_plots_by_dimension(loss_by_group, timestamp)
        for path in saved_loss_paths:
            print(f"Saved loss plot: {path}")

    if SAVE_METRICS_PLOTS:
        saved_metric_plot_paths = _save_metrics_plots(metrics_by_group, timestamp)
        for path in saved_metric_plot_paths:
            print(f"Saved metrics plot: {path}")

    saved_table_paths = _save_metrics_tables(metrics_by_group, timestamp)
    for path in saved_table_paths:
        print(f"Saved table: {path}")

    reports_dir = os.path.join(FRAMEWORK_DIR, "reports")
    return create_experiment_report(all_results, reports_dir)


if __name__ == "__main__":
    report = run()
    print(f"Saved report to: {report}")
