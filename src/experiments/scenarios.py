from __future__ import annotations

import math
from typing import Optional

import torch

from src.flow_matching.model.distribution import GaussianMixtureDistribution

##############################################################
# Base (2D) scenario definitions
##############################################################

# "double_gauss_twice" (2D base)
base_means_double_gauss_twice = {
    "x0": [[-2.0, -2.0], [-2.0, 2.0]],
    "x1": [[2.0, -2.0], [2.0, 2.0]],
}
variances_double_gauss_twice = {"x0": 0.1, "x1": 0.1}

# "gaussian_circles" (2D base)
base_means_gaussian_circles = {
    "x0": [
        [4.0, 0.0],
        [2.8284271, 2.8284271],
        [0.0, 4.0],
        [-2.8284271, 2.8284271],
        [-4.0, 0.0],
        [-2.8284271, -2.8284271],
        [0.0, -4.0],
        [2.8284271, -2.8284271],
    ],
    "x1": [
        [8.0, 0.0],
        [5.6568542, 5.6568542],
        [0.0, 8.0],
        [-5.6568542, 5.6568542],
        [-8.0, 0.0],
        [-5.6568542, -5.6568542],
        [0.0, -8.0],
        [5.6568542, -5.6568542],
    ],
}
variances_gaussian_circles = {"x0": 0.1, "x1": 0.1}

# "double_gauss_twice_uftd" base (2D base x0 + global shift in all dims)
base_means_double_gauss_twice_uftd_x0 = [[0.0, 0.0], [0.0, 4.0]]
shift_double_gauss_twice_uftd = 4.0
variances_double_gauss_twice_uftd = {"x0": 0.1, "x1": 0.1}

# "gaussian_circles_uftd" uses the circles' x0 as base and shifts in all dims
shift_gaussian_circles_uftd = 4.0
variances_gaussian_circles_uftd = {"x0": 0.1, "x1": 0.1}

##############################################################

SCENARIO_NAMES = [
    "double_gauss_twice",
    "double_gauss_twice_ftd",
    "double_gauss_twice_uftd",
    "gaussian_circles",
    "gaussian_circles_ftd",
    "gaussian_circles_uftd",
]


##############################################################
# Helpers
##############################################################

def _embed_2d_center_last(dim: int, xy: list[float]) -> list[float]:
    if dim < 2:
        raise ValueError("dim must be >= 2")
    x, y = xy
    return [0.0] * (dim - 2) + [float(x), float(y)]


def _calculate_normalized_centers_ftd_dg_twice(d: int, mode_sep: float = 2.0):
    """
    Full-transport-dimension (normalized) version:
    - last coord is ±mode_sep
    - first (d-1) coords shift from -a to +a such that ||delta|| = 4
    """
    if d < 2:
        raise ValueError("dim must be >= 2")

    delta = 4.0 / math.sqrt(d - 1)  # per-axis displacement
    a = delta / 2.0

    base_src = [-a] * (d - 1)
    base_tgt = [a] * (d - 1)

    x0_c0 = base_src + [-mode_sep]
    x0_c1 = base_src + [mode_sep]
    x1_c0 = base_tgt + [-mode_sep]
    x1_c1 = base_tgt + [mode_sep]
    return x0_c0, x0_c1, x1_c0, x1_c1


def _calculate_normalized_centers_ftd_gaussian_circles(dim: int) -> tuple[list[list[float]], list[list[float]]]:
    """
    Full-transport-dimension (normalized) version:
    - keep the original 2D circle points in the LAST two dims
    - shift the first (dim-2) dims from -a to +a such that ||delta|| = 4
    """
    if dim < 2:
        raise ValueError("dim must be >= 2")

    transport_dims = dim - 2
    if transport_dims <= 0:
        x0_means = [_embed_2d_center_last(dim, m) for m in base_means_gaussian_circles["x0"]]
        x1_means = [_embed_2d_center_last(dim, m) for m in base_means_gaussian_circles["x1"]]
        return x0_means, x1_means

    delta = 4.0 / math.sqrt(transport_dims)
    a = delta / 2.0
    x0_prefix = [-a] * transport_dims
    x1_prefix = [a] * transport_dims

    x0_means = [x0_prefix + [float(x), float(y)] for x, y in base_means_gaussian_circles["x0"]]
    x1_means = [x1_prefix + [float(x), float(y)] for x, y in base_means_gaussian_circles["x1"]]
    return x0_means, x1_means


def _calculate_centers_n_dim_shift(dim: int, base_means_x0_2d: list[list[float]], shift: float) -> tuple[
    list[list[float]], list[list[float]]]:
    if dim < 2:
        raise ValueError("dim must be >= 2")

    prefix = [0.0] * (dim - 2)
    shift_vector = [float(shift)] * dim

    x0_means: list[list[float]] = [prefix + [float(x), float(y)] for x, y in base_means_x0_2d]
    x1_means: list[list[float]] = [
        [a + b for a, b in zip(shift_vector, m)]  # IMPORTANT: list, not a generator
        for m in x0_means
    ]
    return x0_means, x1_means


##############################################################
# Scenario builder
##############################################################

def _build_scenario_centers_and_w2_sq(
        name: str,
        dim: int,
) -> tuple[list[list[float]], list[list[float]], Optional[float]]:
    """
    Returns (x0_means, x1_means, w2_sq_pre_calc).
    """
    if name == "double_gauss_twice":
        x0_means = [_embed_2d_center_last(dim, m) for m in base_means_double_gauss_twice["x0"]]
        x1_means = [_embed_2d_center_last(dim, m) for m in base_means_double_gauss_twice["x1"]]
        w2_sq_pre_calc = 16.0
        return x0_means, x1_means, w2_sq_pre_calc

    if name == "double_gauss_twice_ftd":
        x0_c0, x0_c1, x1_c0, x1_c1 = _calculate_normalized_centers_ftd_dg_twice(dim)
        x0_means = [x0_c0, x0_c1]
        x1_means = [x1_c0, x1_c1]
        w2_sq_pre_calc = 16.0
        return x0_means, x1_means, w2_sq_pre_calc

    if name == "double_gauss_twice_uftd":
        x0_means, x1_means = _calculate_centers_n_dim_shift(
            dim=dim,
            base_means_x0_2d=base_means_double_gauss_twice_uftd_x0,
            shift=shift_double_gauss_twice_uftd,
        )
        w2_sq_pre_calc = float(dim) * float(shift_double_gauss_twice_uftd) ** 2
        return x0_means, x1_means, w2_sq_pre_calc

    if name == "gaussian_circles":
        x0_means = [_embed_2d_center_last(dim, m) for m in base_means_gaussian_circles["x0"]]
        x1_means = [_embed_2d_center_last(dim, m) for m in base_means_gaussian_circles["x1"]]
        w2_sq_pre_calc = 16.0
        return x0_means, x1_means, w2_sq_pre_calc

    if name == "gaussian_circles_ftd":
        x0_means, x1_means = _calculate_normalized_centers_ftd_gaussian_circles(dim)
        # this is what you had before; keep it as the known pre-calc for that construction
        w2_sq_pre_calc = 32.0
        return x0_means, x1_means, w2_sq_pre_calc

    if name == "gaussian_circles_uftd":
        # Shift *all* embedded coords by (shift,...,shift)
        x0_means, x1_means = _calculate_centers_n_dim_shift(
            dim=dim,
            base_means_x0_2d=base_means_gaussian_circles["x0"],
            shift=shift_gaussian_circles_uftd,
        )
        w2_sq_pre_calc = float(dim) * float(shift_gaussian_circles_uftd) ** 2
        return x0_means, x1_means, w2_sq_pre_calc

    raise ValueError(f"Unknown scenario name: {name}. Available: {SCENARIO_NAMES}")


##############################################################
# Public API
##############################################################

def get_scenario(
        scenario_name: str,
        dim: int,
        device: torch.device,
) -> tuple[GaussianMixtureDistribution, GaussianMixtureDistribution, Optional[float]]:
    """
    Returns (gmd_x0, gmd_x1, w2_sq_pre_calc) for a scenario name.
    """
    x0_means, x1_means, w2_sq_pre_calc = _build_scenario_centers_and_w2_sq(scenario_name, dim)

    if scenario_name in ("double_gauss_twice", "double_gauss_twice_ftd"):
        x0_variance = variances_double_gauss_twice["x0"]
        x1_variance = variances_double_gauss_twice["x1"]
    elif scenario_name == "double_gauss_twice_uftd":
        x0_variance = variances_double_gauss_twice_uftd["x0"]
        x1_variance = variances_double_gauss_twice_uftd["x1"]
    elif scenario_name in ("gaussian_circles", "gaussian_circles_ftd"):
        x0_variance = variances_gaussian_circles["x0"]
        x1_variance = variances_gaussian_circles["x1"]
    elif scenario_name == "gaussian_circles_uftd":
        x0_variance = variances_gaussian_circles_uftd["x0"]
        x1_variance = variances_gaussian_circles_uftd["x1"]
    else:
        raise ValueError(f"Unknown scenario name: {scenario_name}. Available: {SCENARIO_NAMES}")

    gmd_x0 = GaussianMixtureDistribution(means=x0_means, variances=x0_variance, device=device)
    gmd_x1 = GaussianMixtureDistribution(means=x1_means, variances=x1_variance, device=device)

    return gmd_x0, gmd_x1, w2_sq_pre_calc
