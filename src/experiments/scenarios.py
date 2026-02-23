from __future__ import annotations

from typing import Optional

import torch

from src.flow_matching.model.distribution import GaussianMixtureDistribution
from src.flow_matching.controller.utils import calculate_normalized_centers_ftd_dg_twice


# Scenario constants follow the requested naming convention.
means_double_gauss_twice = {
    "x0": [[-2.0, -2.0], [-2.0, 2.0]],
    "x1": [[2.0, -2.0], [2.0, 2.0]],
}
variances_double_gauss_twice = {
    "x0": 0.1,
    "x1": 0.1,
}
full_transport_dim_double_gauss_twice = False
w2_sq_pre_calc_double_gauss_twice: Optional[float] = None

means_double_gauss_twice_ftd = {
    "x0": [[-2.0, -2.0], [-2.0, 2.0]],
    "x1": [[2.0, -2.0], [2.0, 2.0]],
}
variances_double_gauss_twice_ftd = {
    "x0": 0.1,
    "x1": 0.1,
}
full_transport_dim_double_gauss_twice_ftd = True
w2_sq_pre_calc_double_gauss_twice_ftd: Optional[float] = None


SCENARIO_NAMES = [
    "double_gauss_twice",
    "double_gauss_twice_ftd",
]


def _embed_2d_center_last(dim: int, xy: list[float]) -> list[float]:
    if dim < 2:
        raise ValueError("dim must be >= 2")
    x, y = xy
    return [0.0] * (dim - 2) + [float(x), float(y)]


def _build_scenario_centers(name: str, dim: int) -> tuple[list[list[float]], list[list[float]], bool]:
    if name == "double_gauss_twice":
        x0_means = [_embed_2d_center_last(dim, m) for m in means_double_gauss_twice["x0"]]
        x1_means = [_embed_2d_center_last(dim, m) for m in means_double_gauss_twice["x1"]]
        return x0_means, x1_means, full_transport_dim_double_gauss_twice

    if name == "double_gauss_twice_ftd":
        x0_c0, x0_c1, x1_c0, x1_c1 = calculate_normalized_centers_ftd_dg_twice(dim)
        x0_means = [x0_c0, x0_c1]
        x1_means = [x1_c0, x1_c1]
        return x0_means, x1_means, full_transport_dim_double_gauss_twice_ftd

    raise ValueError(f"Unknown scenario name: {name}. Available: {SCENARIO_NAMES}")


def get_scenario(
    scenario_name: str,
    dim: int,
    device: torch.device,
) -> tuple[GaussianMixtureDistribution, GaussianMixtureDistribution, bool, Optional[float]]:
    """
    Returns (gmd_x0, gmd_x1, full_transportation_dim, w2_sq_pre_calc) for a scenario name.
    """
    x0_means, x1_means, full_transportation_dim = _build_scenario_centers(scenario_name, dim)

    if scenario_name == "double_gauss_twice":
        w2_sq_pre_calc = w2_sq_pre_calc_double_gauss_twice
        x0_variance = variances_double_gauss_twice["x0"]
        x1_variance = variances_double_gauss_twice["x1"]
    elif scenario_name == "double_gauss_twice_ftd":
        w2_sq_pre_calc = w2_sq_pre_calc_double_gauss_twice_ftd
        x0_variance = variances_double_gauss_twice_ftd["x0"]
        x1_variance = variances_double_gauss_twice_ftd["x1"]
    else:
        raise ValueError(f"Unknown scenario name: {scenario_name}. Available: {SCENARIO_NAMES}")

    gmd_x0 = GaussianMixtureDistribution(means=x0_means, variances=x0_variance, device=device)
    gmd_x1 = GaussianMixtureDistribution(means=x1_means, variances=x1_variance, device=device)

    return gmd_x0, gmd_x1, full_transportation_dim, w2_sq_pre_calc
