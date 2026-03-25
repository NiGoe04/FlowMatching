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
base_means_dg_twice_dist8 = {
    "x0": [[-4.0, -2.0], [-4.0, 2.0]],
    "x1": [[4.0, -2.0], [4.0, 2.0]],
}
base_means_dg_twice_dist16 = {
    "x0": [[-8.0, -2.0], [-8.0, 2.0]],
    "x1": [[8.0, -2.0], [8.0, 2.0]],
}
base_means_dg_twice_dist32 = {
    "x0": [[-16.0, -2.0], [-16.0, 2.0]],
    "x1": [[16.0, -2.0], [16.0, 2.0]],
}

variances_double_gauss_twice = {"x0": 0.1, "x1": 0.1}

# "gaussian_circles" (2D base)
base_means_gaussian_circles = {
    "x0": [
        [6.0, 0.0],
        [4.2426407, 4.2426407],
        [0.0, 6.0],
        [-4.2426407, 4.2426407],
        [-6.0, 0.0],
        [-4.2426407, -4.2426407],
        [0.0, -6.0],
        [4.2426407, -4.2426407],
    ],
    "x1": [
        [12.0, 0.0],
        [8.4852814, 8.4852814],
        [0.0, 12.0],
        [-8.4852814, 8.4852814],
        [-12.0, 0.0],
        [-8.4852814, -8.4852814],
        [0.0, -12.0],
        [8.4852814, -8.4852814],
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
    "dg_twice_dist8",
    "dg_twice_dist16",
    "dg_twice_dist32",
    "dg_twice_td1_inc_dist",
    "double_gauss_twice_ftd",
    "double_gauss_twice_uftd",
    "gaussian_circles",
    "gaussian_circles_ftd",
    "gaussian_circles_uftd",
    "gaussian_mix_diff_var_1",
    "gaussian_mix_diff_var_2",
    "gaussian_mix_diff_var_3",
]


##############################################################
# Helpers
##############################################################

def _embed_2d_center_last(dim: int, xy: list[float], inc_dist=False, is_source=True) -> list[float]:
    if dim < 2:
        raise ValueError("dim must be >= 2")
    x, y = xy
    if not inc_dist:
        return [0.0] * (dim - 2) + [float(x), float(y)]
    else:
        if is_source:
            return [0.0] * (dim - 2) + [float(x) - dim, float(y)]
        else:
            return [0.0] * (dim - 2) + [float(x) + dim, float(y)]


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

    delta = 6.0 / math.sqrt(transport_dims)
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


def _build_gaussian_mix_diff_var_1(dim: int) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    """
    Build a scalable, non-trivial source/target Gaussian mixture pair with:
      - different number of source/target modes,
      - per-component variance vectors,
      - transport that is not a pure global shift.

    Construction notes:
      - Keep a recognizable 2D signature in the last two dims.
      - Add component-specific structure to the first (dim-2) dims so intrinsic
        dimensionality actually grows with dim.
    """
    if dim < 2:
        raise ValueError("dim must be >= 2")

    x0_xy = [
        [-3.2, -1.0],
        [-1.5, 2.8],
        [0.7, -3.4],
        [2.8, 1.6],
    ]
    x1_xy = [
        [-4.0, 2.5],
        [-0.8, -2.9],
        [2.2, 3.1],
        [4.0, -0.3],
        [0.6, 0.9],
        [3.0, -3.2],
    ]

    extra_dims = dim - 2
    x0_means: list[list[float]] = []
    x1_means: list[list[float]] = []

    for idx, (x, y) in enumerate(x0_xy):
        prefix = [
            0.9 * math.sin(0.45 * (j + 1) + idx * 0.8) + 0.35 * math.cos((idx + 1.0) * (j + 1) / 5.0)
            for j in range(extra_dims)
        ]
        x0_means.append(prefix + [x, y])

    for idx, (x, y) in enumerate(x1_xy):
        prefix = [
            -0.7 * math.cos(0.4 * (j + 1) + idx * 0.55) + 0.55 * math.sin((idx + 1.3) * (j + 1) / 6.5)
            for j in range(extra_dims)
        ]
        x1_means.append(prefix + [x, y])

    x0_variances = [
        [0.030 + 0.010 * ((idx + 2 * j) % 5) for j in range(dim)]
        for idx in range(len(x0_means))
    ]
    x1_variances = [
        [0.040 + 0.012 * ((2 * idx + j) % 6) for j in range(dim)]
        for idx in range(len(x1_means))
    ]

    return x0_means, x1_means, x0_variances, x1_variances


def _build_gaussian_mix_diff_var_2(dim: int) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    """
    Variant of gaussian_mix_diff_var_1 with simple constant prefixes:
      - x0 prefixes are zeros in the first (dim-2) dimensions,
      - x1 prefixes are twos in the first (dim-2) dimensions.
    The 2D signatures and per-component variance construction are identical to
    gaussian_mix_diff_var_1.
    """
    if dim < 2:
        raise ValueError("dim must be >= 2")

    x0_xy = [
        [-3.2, -1.0],
        [-1.5, 2.8],
        [0.7, -3.4],
        [2.8, 1.6],
    ]
    x1_xy = [
        [-4.0, 2.5],
        [-0.8, -2.9],
        [2.2, 3.1],
        [4.0, -0.3],
        [0.6, 0.9],
        [3.0, -3.2],
    ]

    extra_dims = dim - 2
    x0_prefix = [0.0] * extra_dims
    x1_prefix = [2.0] * extra_dims

    x0_means = [x0_prefix + [x, y] for x, y in x0_xy]
    x1_means = [x1_prefix + [x, y] for x, y in x1_xy]

    x0_variances = [
        [0.030 + 0.010 * ((idx + 2 * j) % 5) for j in range(dim)]
        for idx in range(len(x0_means))
    ]
    x1_variances = [
        [0.040 + 0.012 * ((2 * idx + j) % 6) for j in range(dim)]
        for idx in range(len(x1_means))
    ]

    return x0_means, x1_means, x0_variances, x1_variances

def _build_gaussian_mix_diff_var_3(dim: int) -> tuple[
    list[list[float]], list[list[float]], list[list[float]], list[list[float]]
]:
    """
    Build a non-trivial Gaussian mixture pair whose transport difficulty is
    dimension-invariant:

      - source/target are NOT related by one global shift,
      - each component has its own variance vector,
      - higher dimensions contain component-specific structure,
      - but W2^2 does not increase with dim because all added dimensions
        are identical on source and target under the intended correspondence.

    Idea:
      - The last two dims define the actual transport geometry.
      - The first (dim-2) dims are a shared component-specific embedding.
      - Therefore, extra dims add no transport cost.
    """
    if dim < 2:
        raise ValueError("dim must be >= 2")

    # Same number of modes on both sides to make the intended pairing explicit.
    # Transport is non-trivial in the last two dims: definitely not one global shift.
    x0_xy = [
        [-3.2, -1.0],
        [-1.5,  2.8],
        [ 0.7, -3.4],
        [ 2.8,  1.6],
    ]
    x1_xy = [
        [-4.1,  2.3],   # different displacement than others
        [-0.6, -2.7],
        [ 2.4,  3.0],
        [ 3.7, -0.4],
    ]

    extra_dims = dim - 2

    x0_means: list[list[float]] = []
    x1_means: list[list[float]] = []
    x0_variances: list[list[float]] = []
    x1_variances: list[list[float]] = []

    for idx, ((x0, y0), (x1, y1)) in enumerate(zip(x0_xy, x1_xy)):
        # Shared high-dimensional signature for corresponding source/target modes.
        # This makes the scenario nontrivial in high dimension, but cost-free there.
        shared_prefix = [
            0.9 * math.sin(0.45 * (j + 1) + 0.8 * idx)
            + 0.35 * math.cos((idx + 1.1) * (j + 1) / 5.0)
            + 0.15 * math.sin((idx + 2.0) * (j + 1) / 3.7)
            for j in range(extra_dims)
        ]

        x0_means.append(shared_prefix + [x0, y0])
        x1_means.append(shared_prefix + [x1, y1])

        # Component-specific variances, but identical in the added dims on both sides.
        # Hence extra dimensions still contribute zero to W2^2.
        shared_prefix_vars = [
            0.030 + 0.008 * ((idx + 2 * j) % 5)
            for j in range(extra_dims)
        ]

        # Allow different variances in the actual transport plane (last two dims).
        # This keeps the pair nontrivial without making the higher dims harder.
        x0_last2_vars = [
            0.050 + 0.010 * (idx % 3),
            0.070 + 0.012 * ((idx + 1) % 3),
        ]
        x1_last2_vars = [
            0.060 + 0.011 * ((idx + 2) % 3),
            0.045 + 0.014 * (idx % 3),
        ]

        x0_variances.append(shared_prefix_vars + x0_last2_vars)
        x1_variances.append(shared_prefix_vars + x1_last2_vars)

    return x0_means, x1_means, x0_variances, x1_variances


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

    if name == "dg_twice_dist8":
        x0_means = [_embed_2d_center_last(dim, m) for m in base_means_dg_twice_dist8["x0"]]
        x1_means = [_embed_2d_center_last(dim, m) for m in base_means_dg_twice_dist8["x1"]]
        w2_sq_pre_calc = 8.0 ** 2
        return x0_means, x1_means, w2_sq_pre_calc

    if name == "dg_twice_dist16":
        x0_means = [_embed_2d_center_last(dim, m) for m in base_means_dg_twice_dist16["x0"]]
        x1_means = [_embed_2d_center_last(dim, m) for m in base_means_dg_twice_dist16["x1"]]
        w2_sq_pre_calc = 16.0 ** 2
        return x0_means, x1_means, w2_sq_pre_calc

    if name == "dg_twice_dist32":
        x0_means = [_embed_2d_center_last(dim, m) for m in base_means_dg_twice_dist32["x0"]]
        x1_means = [_embed_2d_center_last(dim, m) for m in base_means_dg_twice_dist32["x1"]]
        w2_sq_pre_calc = 32.0 ** 2
        return x0_means, x1_means, w2_sq_pre_calc

    if name == "dg_twice_td1_inc_dist":
        x0_means = [_embed_2d_center_last(dim, m, inc_dist=True, is_source=True) for m in base_means_double_gauss_twice["x0"]]
        x1_means = [_embed_2d_center_last(dim, m, inc_dist=True, is_source=False) for m in base_means_double_gauss_twice["x1"]]
        w2_sq_pre_calc = 16.0 + (2 * dim) ** 2
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
        w2_sq_pre_calc = 72.0
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

    if name == "gaussian_mix_diff_var_1":
        x0_means, x1_means, _, _ = _build_gaussian_mix_diff_var_1(dim)
        return x0_means, x1_means, None

    if name == "gaussian_mix_diff_var_2":
        x0_means, x1_means, _, _ = _build_gaussian_mix_diff_var_2(dim)
        return x0_means, x1_means, None

    if name == "gaussian_mix_diff_var_3":
        x0_means, x1_means, _, _ = _build_gaussian_mix_diff_var_3(dim)
        return x0_means, x1_means, None

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

    if scenario_name in ("double_gauss_twice", "double_gauss_twice_ftd", "dg_twice_td1_inc_dist", "dg_twice_dist8",
                         "dg_twice_dist16", "dg_twice_dist32"):
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
    elif scenario_name == "gaussian_mix_diff_var_1":
        _, _, x0_variance, x1_variance = _build_gaussian_mix_diff_var_1(dim)
    elif scenario_name == "gaussian_mix_diff_var_2":
        _, _, x0_variance, x1_variance = _build_gaussian_mix_diff_var_2(dim)
    elif scenario_name == "gaussian_mix_diff_var_3":
        _, _, x0_variance, x1_variance = _build_gaussian_mix_diff_var_3(dim)
    else:
        raise ValueError(f"Unknown scenario name: {scenario_name}. Available: {SCENARIO_NAMES}")

    gmd_x0 = GaussianMixtureDistribution(means=x0_means, variances=x0_variance, device=device)
    gmd_x1 = GaussianMixtureDistribution(means=x1_means, variances=x1_variance, device=device)

    return gmd_x0, gmd_x1, w2_sq_pre_calc
