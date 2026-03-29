from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def round_to_significant_digit(value: float, significant_digits: int = 1) -> float:
    if value == 0:
        return 0.0
    return float(f"{value:.{significant_digits}g}")


def save_w2_plot(
    output_dir: Path,
    scenario_name: str,
    timestamp: str,
    run_idx: int,
    dims: Iterable[int],
    values_by_ot_batch_size: Dict[int, List[float]],
) -> Path:
    ensure_dir(output_dir)

    sorted_dims = list(dims)
    fig, ax = plt.subplots(figsize=(8, 5))

    for ot_batch_size, values in values_by_ot_batch_size.items():
        ax.plot(sorted_dims, values, marker="o", linewidth=2, label=f"k = {ot_batch_size}")

    ax.set_xlabel("d")
    ax.set_ylabel(r"$W_2^2(p^n, q^n)$")
    ax.set_title(f"{scenario_name} | run {run_idx + 1}")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    filename = f"{scenario_name}_{timestamp}_run_{run_idx + 1}.png"
    file_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    return file_path


def build_w2_latex_table(
    dims: Iterable[int],
    ot_batch_sizes: Iterable[int],
    mean_std_matrix: Dict[int, Dict[int, tuple[float, float]]],
) -> str:
    dims_list = list(dims)
    ot_batch_sizes_list = list(ot_batch_sizes)

    col_spec = "c|" + " ".join(["c"] * len(dims_list))

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "$k \\backslash d$ & " + " & ".join([str(d) for d in dims_list]) + " \\\\",
        "\\midrule",
    ]

    for k in ot_batch_sizes_list:
        row_values = []
        for d in dims_list:
            mean_val, std_val = mean_std_matrix[k][d]
            std_rounded = round_to_significant_digit(std_val, significant_digits=1)
            row_values.append(f"\\num{{{mean_val:.4f} +- {std_rounded}}}")
        lines.append(f"{k} & " + " & ".join(row_values) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "\\caption{Estimated squared 2-Wasserstein distances $W_2^2(p^n, q^n)$ across OT batch sizes $k$ and dimensions $d$. Values are reported as mean $\\pm$ standard deviation over multiple runs.}",
            "\\label{tab:w2_nd_matrix}",
            "",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def save_w2_latex_table(output_dir: Path, scenario_name: str, timestamp: str, latex_content: str) -> Path:
    ensure_dir(output_dir)
    table_path = output_dir / f"{scenario_name}_{timestamp}.tex"
    table_path.write_text(latex_content)
    return table_path
