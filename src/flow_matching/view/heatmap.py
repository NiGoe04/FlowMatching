import matplotlib.pyplot as plt

from src.flow_matching.controller.heatmap import heatmap_tuple_cond_2d
from src.flow_matching.model.distribution import Distribution2D


def visualize_heatmap_tuple_cond_2d(source_dist: Distribution2D,
                                    target_dist: Distribution2D,
                                    num_iterations,
                                    resolution: float,
                                    cmap="plasma",
                                    show_colorbar=True,
                                    figsize=(8, 8),
                                    title="Probability path approximated via heatmap"):
    # call heatmap function
    heatmap, mins, maxs = heatmap_tuple_cond_2d(
        source_dist,
        target_dist,
        num_iterations,
        resolution
    )

    plt.figure(figsize=figsize)
    plt.imshow(
        heatmap,
        origin='lower',
        cmap=cmap,
        interpolation='nearest',
        extent=(mins[0].item(), maxs[0].item(), mins[1].item(), maxs[1].item()),
        aspect='auto'  # ensures scaling matches coordinate ranges
    )
    plt.title(title)

    if show_colorbar:
        plt.colorbar(label="Measure of Overlap")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.show()
