import os
from src.flow_matching.controller.metrics import Metrics
import torch

from src.flow_matching.controller.utils import load_model_n_dim
from src.flow_matching.model.distribution import Distribution
from src.flow_matching.model.losses import TensorCost
from src.flow_matching.shared.md_2d import PARAMS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = "../../models"

model_path_vanilla = os.path.join(MODEL_SAVE_PATH, "model_2D_double_gauss_twice_2026-01-29_16-26-39.pth")
model_path_mac = os.path.join(MODEL_SAVE_PATH, "model_2D_double_gauss_twice_mac_2026-01-29_17-12-25.pth")
model_path_ot_cfm = os.path.join(MODEL_SAVE_PATH, "model_2D_double_gauss_twice_ot_2026-01-30_14-48-05.pth")
model_path_ref_ot_cfm = os.path.join(MODEL_SAVE_PATH, "model_2D_double_gauss_twice_ref_ot_var2_2026-02-02_16-47-15.pth")

model_vanilla = load_model_n_dim(2, model_path_vanilla, device=DEVICE)
model_mac = load_model_n_dim(2, model_path_mac, device=DEVICE)
model_ot_cfm = load_model_n_dim(2, model_path_ot_cfm, device=DEVICE)
model_ref_ot_cfm = load_model_n_dim(2, model_path_ref_ot_cfm, device=DEVICE)

variance_source = 0.1
variance_target = 0.1

x_0_dist_center_0 = [-2, -2]
x_0_dist_center_1 = [-2, 2]
x_1_dist_center_0 = [2, -2]
x_1_dist_center_1 = [2, 2]

x_0_dist_sample_0 = (Distribution(x_0_dist_center_0, int(PARAMS["amount_samples"] / 2), device=DEVICE)
                     .with_gaussian_noise(variance=variance_source))

x_0_dist_sample_1 = (Distribution(x_0_dist_center_1, int(PARAMS["amount_samples"] / 2), device=DEVICE)
                     .with_gaussian_noise(variance=variance_source))

x_0_sample = x_0_dist_sample_0.merged_with(x_0_dist_sample_1).tensor

straightness_vanilla = Metrics.calculate_path_straightness(model_vanilla, x_0_sample)
straightness_ot_cfm = Metrics.calculate_path_straightness(model_ot_cfm, x_0_sample)
straightness_mac = Metrics.calculate_path_straightness(model_mac, x_0_sample)
straightness_ref_ot_cfm = Metrics.calculate_path_straightness(model_ref_ot_cfm, x_0_sample)

print("Vanilla straightness: {}".format(straightness_vanilla))
print("MAC straightness: {}".format(straightness_mac))
print("OT-CFM straightness: {}".format(straightness_ot_cfm))
print("Reflow OT-CFM straightness: {}".format(straightness_ref_ot_cfm))

# expect descending values in order
