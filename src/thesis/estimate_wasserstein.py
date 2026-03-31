from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.experiments.mixed_gaussian_framework.scenarios import get_scenario
from src.flow_matching.model.coupling import Coupler
from src.flow_matching.model.losses import TensorCost
from src.view.utils import build_w2_latex_table, make_timestamp, save_w2_latex_table, save_w2_plot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCENARIO = "gaussian_circles_ftd"
NUM_DATA_POINTS = 80000
OT_BATCH_SIZES = [1, 32, 64, 128, 256, 512]
DIMS = [3, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
ITERATIONS = 6
LOG2_DIM_AXIS = True

PLOTS_OUTPUT_DIR = Path("output/plots")
TABLES_OUTPUT_DIR = Path("output/tables")


def compute_transport_cost(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    return ((x0 - x1) ** 2).sum(dim=1).mean()


def get_distributions(dim: int):
    return get_scenario(SCENARIO, dim, DEVICE)


def estimate_squared_w2(dim: int, ot_batch_size: int) -> float:
    gmd_x0, gmd_x1, _ = get_distributions(dim)
    x0_sample = gmd_x0.sample(NUM_DATA_POINTS)
    x1_sample = gmd_x1.sample(NUM_DATA_POINTS)

    coupler = Coupler(x0_sample, x1_sample)
    coupling = coupler.get_independent_coupling()
    loader = DataLoader(coupling, ot_batch_size, shuffle=True)

    tensor_list_x0 = []
    tensor_list_x1 = []

    for x_0, x_1 in loader:
        batch_size = x_1.shape[0]
        coupler_ot = Coupler(x_0, x_1)
        if ot_batch_size == 1:
            coupling_ot = coupler_ot.get_independent_coupling()
        else:
            coupling_ot = coupler_ot.get_n_ot_coupling(batch_size, TensorCost.quadratic_cost)
        tensor_list_x0.append(coupling_ot.x0)
        tensor_list_x1.append(coupling_ot.x1)

    x0 = torch.cat(tensor_list_x0, dim=0)
    x1 = torch.cat(tensor_list_x1, dim=0)

    perm = torch.randperm(len(x0))
    w2_sq = compute_transport_cost(x0[perm], x1[perm])
    return float(w2_sq.item())


def main() -> None:
    timestamp = make_timestamp()

    results = defaultdict(lambda: defaultdict(list))

    for run_idx in range(ITERATIONS):
        print(f"Run {run_idx + 1}/{ITERATIONS}")
        for ot_batch_size in OT_BATCH_SIZES:
            for dim in DIMS:
                w2_sq = estimate_squared_w2(dim=dim, ot_batch_size=ot_batch_size)
                results[ot_batch_size][dim].append(w2_sq)
                print(f"k={ot_batch_size}, dim={dim}, w2sq={w2_sq:.6f}")

    mean_curves = {}
    for ot_batch_size in OT_BATCH_SIZES:
        mean_curves[ot_batch_size] = []
        for dim in DIMS:
            values = torch.tensor(results[ot_batch_size][dim])
            mean_curves[ot_batch_size].append(float(values.mean().item()))

    plot_path = save_w2_plot(
        output_dir=PLOTS_OUTPUT_DIR,
        scenario_name=SCENARIO,
        timestamp=timestamp,
        dims=DIMS,
        values_by_ot_batch_size=mean_curves,
        log2_dim_axis=LOG2_DIM_AXIS,
    )
    print(f"Saved plot: {plot_path}")

    mean_std_matrix = {}
    for ot_batch_size in OT_BATCH_SIZES:
        mean_std_matrix[ot_batch_size] = {}
        for dim in DIMS:
            values = torch.tensor(results[ot_batch_size][dim])
            mean_std_matrix[ot_batch_size][dim] = (
                float(values.mean().item()),
                float(values.std(unbiased=True).item()),
            )

    latex_content = build_w2_latex_table(
        dims=DIMS,
        ot_batch_sizes=OT_BATCH_SIZES,
        mean_std_matrix=mean_std_matrix,
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
