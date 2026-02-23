import os
import torch

from src.flow_matching.controller.metrics import Metrics
from src.flow_matching.controller.utils import load_model_n_dim
from src.flow_matching.shared.md_2d import PARAMS
from src.flow_matching.model.distribution import GaussianMixtureDistribution

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "../../models"

variance_source = 0.1
variance_target = 0.1

# -------------------------
# Configuration lists (same index = same experiment)
# -------------------------

def embed_2d_center_last(d: int, xy):
    """
    Embed a 2D center (x, y) into R^d by putting it in the last two coordinates.
    Example: d=5, xy=(-2, 2) -> [0, 0, 0, -2, 2]
    """
    if d < 2:
        raise ValueError("d must be >= 2")
    x, y = xy
    return [0.0] * (d - 2) + [float(x), float(y)]


def make_center_lists(dims):
    """
    Builds the same four center-lists you wrote, but automatically from dims.
    """
    # your 2D base centers
    x0_c0_2d = (-2, -2)
    x0_c1_2d = (-2,  2)
    x1_c0_2d = ( 2, -2)
    x1_c1_2d = ( 2,  2)

    x_0_dist_centers_0 = [embed_2d_center_last(d, x0_c0_2d) for d in dims]
    x_0_dist_centers_1 = [embed_2d_center_last(d, x0_c1_2d) for d in dims]
    x_1_dist_centers_0 = [embed_2d_center_last(d, x1_c0_2d) for d in dims]
    x_1_dist_centers_1 = [embed_2d_center_last(d, x1_c1_2d) for d in dims]

    return x_0_dist_centers_0, x_0_dist_centers_1, x_1_dist_centers_0, x_1_dist_centers_1

DIMS = [2, 3, 5, 8, 21, 89, 144, 233, 377, 610]

x_0_dist_centers_0, x_0_dist_centers_1, x_1_dist_centers_0, x_1_dist_centers_1 = make_center_lists(DIMS)

model_paths_vanilla = [
os.path.join(MODEL_SAVE_PATH, "model_2D_double_gauss_twice_2026-01-29_16-26-39.pth"),
os.path.join(MODEL_SAVE_PATH, "model_3D_double_gauss_twice_2026-02-18_12-06-53.pth"),
os.path.join(MODEL_SAVE_PATH, "model_5D_double_gauss_twice_2026-02-18_13-36-49.pth"),
os.path.join(MODEL_SAVE_PATH, "model_8D_double_gauss_twice_2026-02-18_13-49-36.pth"),
os.path.join(MODEL_SAVE_PATH, "model_21D_double_gauss_twice_2026-02-18_14-26-37.pth"),
os.path.join(MODEL_SAVE_PATH, "model_89D_double_gauss_twice_2026-02-20_14-28-21.pth"),
os.path.join(MODEL_SAVE_PATH, "model_144D_double_gauss_twice_2026-02-20_15-20-05.pth"),
os.path.join(MODEL_SAVE_PATH, "model_233D_double_gauss_twice_2026-02-20_16-29-42.pth"),
os.path.join(MODEL_SAVE_PATH, "model_377D_double_gauss_twice_2026-02-20_16-38-50.pth"),
os.path.join(MODEL_SAVE_PATH, "model_610D_double_gauss_twice_2026-02-23_11-36-47.pth"),
]

model_paths_ot_cfm = [
os.path.join(MODEL_SAVE_PATH, "model_2D_double_gauss_twice_ot_2026-01-30_14-48-05.pth"),
os.path.join(MODEL_SAVE_PATH, "model_3D_double_gauss_twice_ot_2026-02-18_12-17-37.pth"),
os.path.join(MODEL_SAVE_PATH, "model_5D_double_gauss_twice_ot_2026-02-18_13-41-46.pth"),
os.path.join(MODEL_SAVE_PATH, "model_8D_double_gauss_twice_ot_2026-02-18_13-53-54.pth"),
os.path.join(MODEL_SAVE_PATH, "model_21D_double_gauss_twice_ot_2026-02-18_14-29-01.pth"),
os.path.join(MODEL_SAVE_PATH, "model_89D_double_gauss_twice_ot_2026-02-20_14-29-47.pth"),
os.path.join(MODEL_SAVE_PATH, "model_144D_double_gauss_twice_ot_2026-02-20_15-22-02.pth"),
os.path.join(MODEL_SAVE_PATH, "model_233D_double_gauss_twice_ot_2026-02-20_16-32-39.pth"),
os.path.join(MODEL_SAVE_PATH, "model_377D_double_gauss_twice_ot_2026-02-20_16-41-09.pth"),
os.path.join(MODEL_SAVE_PATH, "model_610D_double_gauss_twice_ot_2026-02-23_11-38-34.pth"),
]

n_total = int(PARAMS["amount_samples"])

assert len(DIMS) == len(x_0_dist_centers_0) == len(x_0_dist_centers_1) == len(x_1_dist_centers_0) == len(x_1_dist_centers_1) \
       == len(model_paths_vanilla) == len(model_paths_ot_cfm), "Config lists must have the same length!"


def make_two_component_gmm(center0, center1, variance, device):
    """
    Builds a 2-component diagonal GMM with uniform mixture weights.
    - means: [[center0], [center1]]
    - variances: scalar -> broadcasted to [2, d] inside the class
    """
    means = [center0, center1]  # shape [2, d] as list
    return GaussianMixtureDistribution(means=means, variances=variance, device=device)


for i, d in enumerate(DIMS):
    # safety: centers must match d
    if d < 2:
        continue

    assert len(x_0_dist_centers_0[i]) == d
    assert len(x_0_dist_centers_1[i]) == d
    assert len(x_1_dist_centers_0[i]) == d
    assert len(x_1_dist_centers_1[i]) == d

    model_vanilla = load_model_n_dim(d, model_paths_vanilla[i], device=DEVICE)
    model_ot_cfm = load_model_n_dim(d, model_paths_ot_cfm[i], device=DEVICE)

    gmm_x0 = make_two_component_gmm(
        x_0_dist_centers_0[i], x_0_dist_centers_1[i],
        variance=variance_source,
        device=DEVICE
    )
    gmm_x1 = make_two_component_gmm(
        x_1_dist_centers_0[i], x_1_dist_centers_1[i],
        variance=variance_target,
        device=DEVICE
    )

    x0 = gmm_x0.sample(n_total)
    x1_gt = gmm_x1.sample(n_total)

    # Straightness
    straight_v = Metrics.calculate_path_straightness(model_vanilla, x0)
    straight_o = Metrics.calculate_path_straightness(model_ot_cfm, x0)

    # NPE returns (PE, W2^2, NPE)
    pe_v, w2_v, npe_v = Metrics.calculate_normalized_path_energy(model_vanilla, x0, x1_gt)
    pe_o, w2_o, npe_o = Metrics.calculate_normalized_path_energy(model_ot_cfm, x0, x1_gt)

    # NLL
    _, psi_1_v = Metrics._calculate_mean_velocity_norm_sq(model_vanilla, x0)
    _, psi_1_o = Metrics._calculate_mean_velocity_norm_sq(model_ot_cfm, x0)

    nll_v = gmm_x1.nll(psi_1_v)
    nll_mi_corr_v = gmm_x1.nll_mi_corrected(psi_1_v)
    nll_o = gmm_x1.nll(psi_1_o)
    nll_mi_corr_o = gmm_x1.nll_mi_corrected(psi_1_o)

    nll_per_dim_v = gmm_x1.nll_per_dim(psi_1_v)
    nll_mi_corr_per_dim_v = gmm_x1.nll_mi_corrected_per_dim(psi_1_v)
    nll_per_dim_o = gmm_x1.nll_per_dim(psi_1_o)
    nll_mi_corr_per_dim_o = gmm_x1.nll_mi_corrected_per_dim(psi_1_o)

    print(f"========== d = {d} ==========")
    print(f"vanilla straightness:             {straight_v.item()}")
    print(f"OT-CFM  straightness:             {straight_o.item()}")
    print(f"vanilla PE:                       {pe_v.item()}")
    print(f"OT-CFM  PE:                       {pe_o.item()}")
    print(f"Est. vanilla W2^2:                {w2_v.item()}")
    print(f"Est. OT-CFM  W2^2:                {w2_o.item()}")
    print(f"vanilla NPE:                      {npe_v.item()}")
    print(f"OT-CFM  NPE:                      {npe_o.item()}")
    print()
    print(f"vanilla NLL:                      {nll_v.item()}")
    print(f"vanilla MI-corrected NLL:         {nll_mi_corr_v.item()}")
    print(f"OT-CFM NLL:                       {nll_o.item()}")
    print(f"OT-CFM MI-corrected NLL:          {nll_mi_corr_o.item()}")
    print(f"vanilla NLL per dim:              {nll_per_dim_v.item()}")
    print(f"vanilla MI-corrected NLL per dim: {nll_mi_corr_per_dim_v.item()}")
    print(f"OT-CFM NLL per dim:               {nll_per_dim_o.item()}")
    print(f"OT-CFM MI-corrected NLL per dim:  {nll_mi_corr_per_dim_o.item()}")
    print()