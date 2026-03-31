from __future__ import annotations

import itertools
from collections import defaultdict

import matplotlib.pyplot as plt
import torch

from src.experiments.imagenet_framework.create_report import create_experiment_report
from src.experiments.imagenet_framework.mass_training_imagenet import train_or_get_model
from src.experiments.imagenet_framework.utils import (
    LOSS_PLOTS_DIR,
    METRICS_PLOTS_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    load_imagenet_scenario_data,
    sample_model_images,
)
from src.flow_matching.controller.metrics import Metrics
from src.flow_matching.controller.utils import load_model_unet_imagenet
from src.view.utils import decimal_places, ensure_dir, make_timestamp, round_to_significant_digit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_LOSS_PLOTS = True
SAVE_METRICS_PLOTS = True
SAVE_METRICS_TABLES = True
ITERATIONS = 3
LOG2_DIM_AXIS = True

DIMS = [8, 16, 32, 64]
OT_BATCH_SIZES = [1, 32, 64, 128, 256]
OT_OPTIMIZERS = ["hungarian"]
SINKHORN_EPSILONS = [0.01, 0.1]

PARAMS_EXP = {
    "num_epochs": 10,
    "learning_rate": 1.8e-3,
    "dropout_rate_model": 0.0,
    "size_train_set": 40000,
    "amount_samples": 1000,
    "num_trainer_val_samples": 1000,
    "solver_steps": 100,
    "t_end": 1.0,
}

METRIC_KEYS = ["path_energy", "normalized_path_energy", "straightness", "fid", "is"]
METRIC_LATEX_LABELS = {
    "path_energy": r"$\mathrm{PE}(\theta)$",
    "normalized_path_energy": r"$\mathrm{NPE}(\theta)$",
    "straightness": r"$S(\theta)$",
    "fid": r"$\mathrm{FID}(\theta)$",
    "is": r"$\mathrm{IS}(\theta)$",
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
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return out


def _mean_and_unbiased_std(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    mean_val = float(tensor.mean().item())
    if tensor.numel() < 2:
        return mean_val, 0.0
    return mean_val, float(tensor.std(unbiased=True).item())


def _compute_metrics(model, real_images: torch.Tensor, dim: int) -> dict[str, float]:
    amount_samples = int(PARAMS_EXP["amount_samples"])
    x1_eval = real_images[:amount_samples].to(DEVICE)
    x0_eval = torch.randn_like(x1_eval, device=DEVICE)

    straightness = Metrics.calculate_path_straightness(model, x0_eval)
    pe, _, npe = Metrics.calculate_normalized_path_energy(model, x0_eval, x1_eval)

    generated = sample_model_images(
        model=model,
        dim=dim,
        amount_samples=amount_samples,
        solver_steps=int(PARAMS_EXP["solver_steps"]),
        t_end=float(PARAMS_EXP["t_end"]),
        device=DEVICE,
    )
    real_eval = real_images[:amount_samples].to(DEVICE)
    fid = Metrics.calculate_fid(real_eval, generated)
    inception_score, _ = Metrics.calculate_inception_score(generated)

    return {
        "straightness": float(straightness.item()),
        "path_energy": float(pe.item()),
        "normalized_path_energy": float(npe.item()),
        "fid": float(fid.item()),
        "is": float(inception_score.item()),
    }


def _save_loss_plots(loss_by_k: dict[int, dict[str, dict[int, list[float]]]], timestamp: str) -> list[str]:
    ensure_dir(LOSS_PLOTS_DIR)
    paths = []
    for ot_batch_size, losses_by_kind in sorted(loss_by_k.items()):
        for loss_kind, losses_by_dim in losses_by_kind.items():
            if not losses_by_dim:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for dim in sorted(losses_by_dim):
                vals = losses_by_dim[dim]
                ax.plot(range(1, len(vals) + 1), vals, linewidth=2, label=f"d={dim}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.2)
            ax.legend()
            out_path = LOSS_PLOTS_DIR / f"imagenet_k_{ot_batch_size}_{loss_kind}_{timestamp}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            paths.append(str(out_path))
    return paths


def _save_metric_plots(metrics_by_group: dict[tuple[int, int], dict[str, list[float]]], timestamp: str) -> list[str]:
    ensure_dir(METRICS_PLOTS_DIR)
    paths = []
    dims = sorted({k[0] for k in metrics_by_group})
    ks = sorted({k[1] for k in metrics_by_group})

    for metric_key in METRIC_KEYS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for ot_batch_size in ks:
            mean_values = []
            for dim in dims:
                vals = metrics_by_group[(dim, ot_batch_size)][metric_key]
                mean_values.append(_mean_and_unbiased_std(vals)[0])
            ax.plot(dims, mean_values, marker="o", linewidth=2, label=f"k = {ot_batch_size}")
        ax.set_xlabel("d")
        ax.set_ylabel(METRIC_LATEX_LABELS[metric_key])
        if LOG2_DIM_AXIS:
            ax.set_xscale("log", base=2)
            ax.set_xticks(dims)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(alpha=0.2)
        ax.legend()
        out_path = METRICS_PLOTS_DIR / f"imagenet_metric_{metric_key}_{timestamp}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        paths.append(str(out_path))
    return paths


def _build_metrics_table(ot_batch_size: int, dims: list[int], metrics_by_group: dict[tuple[int, int], dict[str, list[float]]]) -> str:
    def fmt(mean_val: float, std_val: float) -> str:
        std_rounded = round_to_significant_digit(std_val, significant_digits=1)
        decimals = decimal_places(std_rounded)
        mean_rounded = round(mean_val, decimals)
        return f"\\num{{{mean_rounded:.{decimals}f} +- {std_rounded:.{decimals}f}}}"

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "",
        "\\begin{tabular}{c|c c c c c}",
        "\\toprule",
        "$d$ & $\\mathrm{PE}(\\theta)$ & $\\mathrm{NPE}(\\theta)$ & $S(\\theta)$ & $\\mathrm{FID}(\\theta)$ & $\\mathrm{IS}(\\theta)$ " + r"\\",
        "\\midrule",
    ]

    for dim in dims:
        stats = {m: _mean_and_unbiased_std(metrics_by_group[(dim, ot_batch_size)][m]) for m in METRIC_KEYS}
        lines.append(
            f"{dim}"
            f" & {fmt(*stats['path_energy'])}"
            f" & {fmt(*stats['normalized_path_energy'])}"
            f" & {fmt(*stats['straightness'])}"
            f" & {fmt(*stats['fid'])}"
            f" & {fmt(*stats['is'])} "
            + r"\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "",
        f"\\caption{{ImageNet metrics across dimensions for OT batch size $k={ot_batch_size}$.}}",
        f"\\label{{tab:imagenet_metrics_k_{ot_batch_size}}}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def _save_metric_tables(metrics_by_group: dict[tuple[int, int], dict[str, list[float]]], timestamp: str) -> list[str]:
    ensure_dir(TABLES_DIR)
    paths = []
    dims = sorted({k[0] for k in metrics_by_group})
    ks = sorted({k[1] for k in metrics_by_group})
    for ot_batch_size in ks:
        content = _build_metrics_table(ot_batch_size, dims, metrics_by_group)
        out_path = TABLES_DIR / f"imagenet_k_{ot_batch_size}_{timestamp}.tex"
        out_path.write_text(content, encoding="utf-8")
        paths.append(str(out_path))
    return paths


def run() -> str:
    timestamp = make_timestamp()

    all_results: list[dict] = []
    metrics_by_group: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    loss_by_group: dict[int, dict[str, dict[int, list[float]]]] = defaultdict(lambda: {"train": {}, "val": {}})

    for dim, ot_batch_size, (ot_optimizer, epsilon) in itertools.product(DIMS, OT_BATCH_SIZES, _optimizer_and_epsilon_grid()):
        print(f"[SCENARIO] imagenet_{dim} | k={ot_batch_size} | opt={ot_optimizer} | eps={epsilon}")
        real_images = load_imagenet_scenario_data(dim, int(PARAMS_EXP["size_train_set"]), DEVICE)
        x1_train = real_images
        x0_train = torch.randn_like(x1_train, device=DEVICE)

        model_path, train_losses, val_losses, trained = train_or_get_model(
            dim=dim,
            x0_train=x0_train,
            x1_train=x1_train,
            ot_batch_size=ot_batch_size,
            ot_optimizer=ot_optimizer,
            epsilon=epsilon,
            params_exp=PARAMS_EXP,
            device=DEVICE,
        )

        if SAVE_LOSS_PLOTS and trained and train_losses and val_losses:
            loss_by_group[ot_batch_size]["train"][dim] = train_losses
            loss_by_group[ot_batch_size]["val"][dim] = val_losses

        model = load_model_unet_imagenet(
            model_path=model_path,
            img_size=dim,
            dropout_rate=float(PARAMS_EXP["dropout_rate_model"]),
            device=DEVICE,
        )
        for _ in range(ITERATIONS):
            vals = _compute_metrics(model, real_images, dim)
            for metric_name, metric_value in vals.items():
                metrics_by_group[(dim, ot_batch_size)][metric_name].append(metric_value)

        metrics_mean = {m: _mean_and_unbiased_std(metrics_by_group[(dim, ot_batch_size)][m])[0] for m in METRIC_KEYS}
        metrics_std = {f"{m}_std": _mean_and_unbiased_std(metrics_by_group[(dim, ot_batch_size)][m])[1] for m in METRIC_KEYS}

        all_results.append({
            "combination": {
                "dim": dim,
                "ot_batch_size": ot_batch_size,
                "ot_optimizer": ot_optimizer,
                "epsilon": epsilon,
                "model_path": model_path,
            },
            "metrics": {**metrics_mean, **metrics_std},
        })

    if SAVE_LOSS_PLOTS:
        for p in _save_loss_plots(loss_by_group, timestamp):
            print(f"Saved loss plot: {p}")
    if SAVE_METRICS_PLOTS:
        for p in _save_metric_plots(metrics_by_group, timestamp):
            print(f"Saved metrics plot: {p}")
    if SAVE_METRICS_TABLES:
        for p in _save_metric_tables(metrics_by_group, timestamp):
            print(f"Saved metrics table: {p}")

    return create_experiment_report(all_results, str(REPORTS_DIR))


if __name__ == "__main__":
    report = run()
    print(f"Saved report to: {report}")
