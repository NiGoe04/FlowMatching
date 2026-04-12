from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch

from src.experiments.mixed_gaussian_framework.scenarios import get_scenario
from src.flow_matching.model.coupling import Coupler
from src.flow_matching.model.losses import TensorCost
from src.view.utils import build_w2_latex_table, make_timestamp, save_w2_latex_table, save_w2_plot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCENARIO = "tri_gauss_twice_ftd"
NUM_DATA_POINTS = [2000] #[100, 500, 1000, 2000]
DIMS = [3, 16, 256, 512, 1024, 2048] #[3, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
ITERATIONS = 1
SAVE_RESULTS = False
LOG2_DIM_AXIS = True

PLOTS_OUTPUT_DIR = Path("output/v2/plots")
TABLES_OUTPUT_DIR = Path("output/v2/tables")


def compute_transport_cost(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    return ((x0 - x1) ** 2).sum(dim=1).mean()


def get_distributions(dim: int):
    return get_scenario(SCENARIO, dim, DEVICE)


def estimate_squared_w2(dim: int, num_data_points: int) -> float:
    gmd_x0, gmd_x1, _ = get_distributions(dim)
    x0_sample = gmd_x0.sample(num_data_points)
    x1_sample = gmd_x1.sample(num_data_points)

    coupler_ot = Coupler(x0_sample, x1_sample)
    coupling_ot = coupler_ot.get_n_ot_coupling(num_data_points, TensorCost.quadratic_cost)

    perm = torch.randperm(len(coupling_ot.x0))
    w2_sq = compute_transport_cost(coupling_ot.x0[perm], coupling_ot.x1[perm])
    return float(w2_sq.item())


def main() -> None:
    timestamp = make_timestamp()

    results = defaultdict(lambda: defaultdict(list))

    for run_idx in range(ITERATIONS):
        print(f"Run {run_idx + 1}/{ITERATIONS}")
        for num_data_points in NUM_DATA_POINTS:
            for dim in DIMS:
                w2_sq = estimate_squared_w2(dim=dim, num_data_points=num_data_points)
                results[num_data_points][dim].append(w2_sq)
                print(f"n={num_data_points}, dim={dim}, w2sq={w2_sq:.6f}")

    mean_curves = {}
    for num_data_points in NUM_DATA_POINTS:
        mean_curves[num_data_points] = []
        for dim in DIMS:
            values = torch.tensor(results[num_data_points][dim])
            mean_curves[num_data_points].append(float(values.mean().item()))

    if SAVE_RESULTS:
        plot_path = save_w2_plot(
            output_dir=PLOTS_OUTPUT_DIR,
            scenario_name=SCENARIO,
            timestamp=timestamp,
            dims=DIMS,
            values_by_ot_batch_size=mean_curves,
            log2_dim_axis=LOG2_DIM_AXIS,
            use_num_data_points_symbol=True,
        )
        print(f"Saved plot: {plot_path}")

    mean_std_matrix = {}
    for num_data_points in NUM_DATA_POINTS:
        mean_std_matrix[num_data_points] = {}
        for dim in DIMS:
            values = torch.tensor(results[num_data_points][dim])
            mean_std_matrix[num_data_points][dim] = (
                float(values.mean().item()),
                float(values.std(unbiased=True).item()),
            )

    if SAVE_RESULTS:
        latex_content = build_w2_latex_table(
            dims=DIMS,
            ot_batch_sizes=NUM_DATA_POINTS,
            mean_std_matrix=mean_std_matrix,
            use_num_data_points_symbol=True,
        )
        table_path = save_w2_latex_table(
            output_dir=TABLES_OUTPUT_DIR,
            scenario_name=SCENARIO,
            timestamp=timestamp,
            latex_content=latex_content,
        )
        print(f"Saved table: {table_path}")


if __name__ == "__main__":
    main()
