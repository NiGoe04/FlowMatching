import torch

from src.flow_matching.model.distribution import Distribution2D
from src.flow_matching.model.losses import TensorCost
from src.flow_matching.view.heatmap import visualize_heatmap_minibatch_ot_2d


# Scenario parameters
N_SAMPLES = 1000
VARIANCE_SOURCE = 60
VARIANCE_TARGET = 60
BOUNDS = [-4, 4, -16, 16]

# Heatmap parameters (requested)
RESOLUTION = 10
OT_BATCH_SIZE = 10
OT_ITERATIONS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


source = Distribution2D([-4, 0], int(N_SAMPLES), device=DEVICE).with_gaussian_noise_y(
    variance=VARIANCE_SOURCE
)

target = Distribution2D([4, 0], int(N_SAMPLES), device=DEVICE).with_gaussian_noise_y(
    variance=VARIANCE_TARGET
)

visualize_heatmap_minibatch_ot_2d(
    source_dist=source,
    target_dist=target,
    num_ot_batch_size=OT_BATCH_SIZE,
    ot_cost_fn=TensorCost.quadratic_cost,
    num_iterations=OT_ITERATIONS,
    resolution=RESOLUTION,
    bounds=BOUNDS
)
