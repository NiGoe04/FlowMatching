import torch

from src.experiments.scenarios import get_scenario
from src.flow_matching.view.utils import plot_tensor_2d, plot_tensor_3d


SCENARIO = "gaussian_mix_diff_var_1"
N_SAMPLES = 3_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sample_combined(dim: int) -> torch.Tensor:
    gmd_x0, gmd_x1, _ = get_scenario(SCENARIO, dim=dim, device=DEVICE)
    x0 = gmd_x0.sample(N_SAMPLES)
    x1 = gmd_x1.sample(N_SAMPLES)
    return torch.cat((x0, x1), dim=0)


def main():
    points_2d = _sample_combined(dim=2)
    plot_tensor_2d(
        points_2d[:, :2],
        title=f"{SCENARIO}: x0 + x1 (2D)",
        params={"scenario": SCENARIO, "dim": 2, "n_per_dist": N_SAMPLES},
    )

    points_3d = _sample_combined(dim=3)
    plot_tensor_3d(
        points_3d[:, :3],
        title=f"{SCENARIO}: x0 + x1 (3D)",
        params={"scenario": SCENARIO, "dim": 3, "n_per_dist": N_SAMPLES},
    )


if __name__ == "__main__":
    main()
