import torch

from src.experiments.mixed_gaussian_framework.scenarios import get_scenario
from src.flow_matching.controller.metrics import Metrics
from src.flow_matching.view.utils import plot_tensor_2d, plot_tensor_3d


SCENARIO = "gaussian_mix_diff_var_3"
N_SAMPLES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sample_combined(dim: int):
    gmd_x0, gmd_x1, _ = get_scenario(SCENARIO, dim=dim, device=DEVICE)
    x0 = gmd_x0.sample(N_SAMPLES)
    x1 = gmd_x1.sample(N_SAMPLES)
    if dim == 2:
        plot_tensor_2d(x0)
        plot_tensor_2d(x1)
    return torch.cat((x0, x1), dim=0), x0, x1


def main():
    points_2d, x0_sample, x1_sample = _sample_combined(dim=2)
    plot_tensor_2d(
        points_2d[:, :2],
        title=f"{SCENARIO}: x0 + x1 (2D)",
        params={"scenario": SCENARIO, "dim": 2, "n_per_dist": N_SAMPLES},
    )
    w2_sq = Metrics.estimate_w2_sq(x0_sample, x1_sample)
    print("dim: {}, w2sq: {}".format(2, w2_sq.item()))

    points_3d, x0_sample, x1_sample = _sample_combined(dim=3)
    plot_tensor_3d(
        points_3d[:, :3],
        title=f"{SCENARIO}: x0 + x1 (3D)",
        params={"scenario": SCENARIO, "dim": 3, "n_per_dist": N_SAMPLES},
    )
    w2_sq = Metrics.estimate_w2_sq(x0_sample, x1_sample)
    print("dim: {}, w2sq: {}".format(3, w2_sq.item()))

    points_4d, x0_sample, x1_sample = _sample_combined(dim=3)
    w2_sq = Metrics.estimate_w2_sq(x0_sample, x1_sample)
    print("dim: {}, w2sq: {}".format(4, w2_sq.item()))


if __name__ == "__main__":
    main()
