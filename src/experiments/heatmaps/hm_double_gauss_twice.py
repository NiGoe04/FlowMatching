import torch

from src.flow_matching.model.distribution import Distribution2D
from src.flow_matching.model.losses import TensorCost
from src.flow_matching.view.heatmap import visualize_heatmap_minibatch_ot_2d


# Scenario parameters
N_SAMPLES = 1000
VARIANCE_SOURCE = 0.1
VARIANCE_TARGET = 0.1
BOUNDS = [-2, 2, -4, 4]

# Heatmap parameters (requested)
RESOLUTION = 20
OT_BATCH_SIZE = 1
OT_ITERATIONS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x0_mode_0 = Distribution2D([-2, -2], int(N_SAMPLES / 2), device=DEVICE).with_gaussian_noise_y(
    variance=VARIANCE_SOURCE
)
x0_mode_1 = Distribution2D([-2, 2], int(N_SAMPLES / 2), device=DEVICE).with_gaussian_noise_y(
    variance=VARIANCE_SOURCE
)
x1_mode_0 = Distribution2D([2, -2], int(N_SAMPLES / 2), device=DEVICE).with_gaussian_noise_y(
    variance=VARIANCE_TARGET
)
x1_mode_1 = Distribution2D([2, 2], int(N_SAMPLES / 2), device=DEVICE).with_gaussian_noise_y(
    variance=VARIANCE_TARGET
)

source = x0_mode_0.merged_with(x0_mode_1)
target = x1_mode_0.merged_with(x1_mode_1)


visualize_heatmap_minibatch_ot_2d(
    source_dist=source,
    target_dist=target,
    num_ot_batch_size=OT_BATCH_SIZE,
    ot_cost_fn=TensorCost.quadratic_cost,
    num_iterations=OT_ITERATIONS,
    resolution=RESOLUTION,
    bounds=BOUNDS
)
