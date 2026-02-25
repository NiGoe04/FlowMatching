from __future__ import annotations

import math
from typing import Optional

import torch

from src.flow_matching.model.distribution import GaussianMixtureDistribution


# Scenario constants follow the requested naming convention.

##############################################################

### "double_gauss_twice"
means_double_gauss_twice = {
    "x0": [[-2.0, -2.0], [-2.0, 2.0]],
    "x1": [[2.0, -2.0], [2.0, 2.0]],
}
variances_double_gauss_twice = {
    "x0": 0.1,
    "x1": 0.1,
}
full_transport_dim_double_gauss_twice = False
w2_sq_pre_calc_double_gauss_twice: Optional[float] = 16.0

### "double_gauss_twice_ftd"
means_double_gauss_twice_ftd = {
    "x0": [[-2.0, -2.0], [-2.0, 2.0]],
    "x1": [[2.0, -2.0], [2.0, 2.0]],
}
variances_double_gauss_twice_ftd = {
    "x0": 0.1,
    "x1": 0.1,
}
full_transport_dim_double_gauss_twice_ftd = True
w2_sq_pre_calc_double_gauss_twice_ftd: Optional[float] = 16.0

### "gaussian_circles"
means_gaussian_circles = {
    "x0": [[4.0, 0.0],
           [2.8284271, 2.8284271],
           [0.0, 4.0],
           [-2.8284271, 2.8284271],
           [-4.0, 0.0],
           [-2.8284271, -2.8284271],
           [0.0, -4.0],
           [2.8284271, -2.8284271]],
    "x1": [[ 8.0,  0.0],
           [ 5.6568542,  5.6568542],
           [ 0.0,  8.0],
           [-5.6568542,  5.6568542],
           [-8.0,  0.0],
           [-5.6568542, -5.6568542],
           [ 0.0, -8.0],
           [ 5.6568542, -5.6568542]],
}

variances_gaussian_circles = {
    "x0": 0.1,
    "x1": 0.1,
}
full_transport_dim_gaussian_circles = False
w2_sq_pre_calc_gaussian_circles: Optional[float] = 16.0

### "gaussian_circles_ftd"
means_gaussian_circles_ftd = means_gaussian_circles
variances_gaussian_circles_ftd = {
    "x0": 0.1,
    "x1": 0.1,
}
full_transport_dim_gaussian_circles_ftd = True
w2_sq_pre_calc_gaussian_circles_ftd: Optional[float] = 32.0

##############################################################

SCENARIO_NAMES = [
    "double_gauss_twice",
    "double_gauss_twice_ftd",
    "gaussian_circles",
    "gaussian_circles_ftd",
]

def _calculate_normalized_centers_ftd_dg_twice(d: int, mode_sep: float = 2.0):
    assert d >= 2
    delta = 4.0 / math.sqrt(d - 1)   # per-axis displacement
    a = delta / 2.0

    base_src = [-a] * (d - 1)
    base_tgt = [ a] * (d - 1)

    x0_c0 = base_src + [-mode_sep]
    x0_c1 = base_src + [ mode_sep]
    x1_c0 = base_tgt + [-mode_sep]
    x1_c1 = base_tgt + [ mode_sep]
    return x0_c0, x0_c1, x1_c0, x1_c1

def _embed_2d_center_last(dim: int, xy: list[float]) -> list[float]:
    if dim < 2:
        raise ValueError("dim must be >= 2")
    x, y = xy
    return [0.0] * (dim - 2) + [float(x), float(y)]


def _calculate_normalized_centers_ftd_gaussian_circles(dim: int) -> tuple[list[list[float]], list[list[float]]]:
    if dim < 2:
        raise ValueError("dim must be >= 2")

    transport_dims = dim - 2
    if transport_dims <= 0:
        x0_means = [_embed_2d_center_last(dim, m) for m in means_gaussian_circles_ftd["x0"]]
        x1_means = [_embed_2d_center_last(dim, m) for m in means_gaussian_circles_ftd["x1"]]
        return x0_means, x1_means

    delta = 4.0 / math.sqrt(transport_dims)
    a = delta / 2.0
    x0_prefix = [-a] * transport_dims
    x1_prefix = [a] * transport_dims

    x0_means = [x0_prefix + [float(x), float(y)] for x, y in means_gaussian_circles_ftd["x0"]]
    x1_means = [x1_prefix + [float(x), float(y)] for x, y in means_gaussian_circles_ftd["x1"]]
    return x0_means, x1_means


def _build_scenario_centers(name: str, dim: int) -> tuple[list[list[float]], list[list[float]], bool]:
    if name == "double_gauss_twice":
        x0_means = [_embed_2d_center_last(dim, m) for m in means_double_gauss_twice["x0"]]
        x1_means = [_embed_2d_center_last(dim, m) for m in means_double_gauss_twice["x1"]]
        return x0_means, x1_means, full_transport_dim_double_gauss_twice

    if name == "double_gauss_twice_ftd":
        x0_c0, x0_c1, x1_c0, x1_c1 = _calculate_normalized_centers_ftd_dg_twice(dim)
        x0_means = [x0_c0, x0_c1]
        x1_means = [x1_c0, x1_c1]
        return x0_means, x1_means, full_transport_dim_double_gauss_twice_ftd

    if name == "gaussian_circles":
        x0_means = [_embed_2d_center_last(dim, m) for m in means_gaussian_circles["x0"]]
        x1_means = [_embed_2d_center_last(dim, m) for m in means_gaussian_circles["x1"]]
        return x0_means, x1_means, full_transport_dim_gaussian_circles

    if name == "gaussian_circles_ftd":
        x0_means, x1_means = _calculate_normalized_centers_ftd_gaussian_circles(dim)
        return x0_means, x1_means, full_transport_dim_gaussian_circles_ftd

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
    elif scenario_name == "gaussian_circles":
        w2_sq_pre_calc = w2_sq_pre_calc_gaussian_circles
        x0_variance = variances_gaussian_circles["x0"]
        x1_variance = variances_gaussian_circles["x1"]
    elif scenario_name == "gaussian_circles_ftd":
        w2_sq_pre_calc = w2_sq_pre_calc_gaussian_circles_ftd
        x0_variance = variances_gaussian_circles_ftd["x0"]
        x1_variance = variances_gaussian_circles_ftd["x1"]
    else:
        raise ValueError(f"Unknown scenario name: {scenario_name}. Available: {SCENARIO_NAMES}")

    gmd_x0 = GaussianMixtureDistribution(means=x0_means, variances=x0_variance, device=device)
    gmd_x1 = GaussianMixtureDistribution(means=x1_means, variances=x1_variance, device=device)

    return gmd_x0, gmd_x1, full_transportation_dim, w2_sq_pre_calc
