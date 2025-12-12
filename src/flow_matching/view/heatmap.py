import matplotlib.pyplot as plt

from src.flow_matching.controller.heatmap import heatmap_tuple_cond_2d
from src.flow_matching.model.distribution import Distribution2D


def visualize_heatmap_tuple_cond_2d(source_dist: Distribution2D,
                                    target_dist: Distribution2D,
                                    num_iterations,
                                    grid_granularity,
                                    cmap="hot",
                                    show_colorbar=True,
                                    figsize=(8, 8),
                                    title="Heatmap of Coupled Lines"):
    heatmap = heatmap_tuple_cond_2d(
        source_dist,
        target_dist,
        num_iterations,
        grid_granularity
    )
    plt.figure(figsize=figsize)
    plt.imshow(heatmap, origin='lower', cmap=cmap, interpolation='nearest')
    plt.title(title)

    if show_colorbar:
        plt.colorbar(label="Line overlap count")

    plt.xlabel("x index")
    plt.ylabel("y index")

    plt.tight_layout()
    plt.show()
