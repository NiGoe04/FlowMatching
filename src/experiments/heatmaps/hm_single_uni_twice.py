import torch

from src.flow_matching.model.distribution import Distribution2D
from src.flow_matching.model.losses import TensorCost
from src.flow_matching.view.heatmap import visualize_heatmap_minibatch_ot_2d


# Scenario parameters
N_SAMPLES = 1000
noise_bound_source = 6
noise_bound_target = 6
BOUNDS = [-4, 4, -8, 8]

# Heatmap parameters (requested)
RESOLUTION = 12
OT_BATCH_SIZE = 1
OT_ITERATIONS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


source = Distribution2D([-4, 0], int(N_SAMPLES), device=DEVICE).with_uniform_noise_y(
    noise_bound=noise_bound_source
)

target = Distribution2D([4, 0], int(N_SAMPLES), device=DEVICE).with_uniform_noise_y(
    noise_bound=noise_bound_target
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
