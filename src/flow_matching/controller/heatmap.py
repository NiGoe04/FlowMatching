import numpy as np
import torch

from src.flow_matching.model.coupling import Coupler
from src.flow_matching.model.distribution import Distribution2D

def bresenham(x0, y0, x1, y1):
    points = []

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx + dy
    x, y = x0, y0

    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

    return points

def heatmap_tuple_cond_2d(source_dist: Distribution2D,
                          target_dist: Distribution2D,
                          num_iterations,
                          grid_granularity):
    # create heatmap grid
    all_points = torch.cat([source_dist.tensor, target_dist.tensor], dim=0)

    mins = all_points.min(dim=0).values  # shape [2]
    maxs = all_points.max(dim=0).values  # shape [2]

    padding = grid_granularity
    mins -= padding
    maxs += padding

    def coord_to_idx(coord, min_val):
        return ((coord - min_val) / grid_granularity).round().long()

    width = coord_to_idx(maxs[0], mins[0]).item() + 1
    height = coord_to_idx(maxs[1], mins[1]).item() + 1

    heatmap = np.zeros((height, width), dtype=np.int32)

    # fill heatmap
    coupler = Coupler(source_dist.tensor, target_dist.tensor)

    for _ in range(num_iterations):
        coupling = coupler.get_independent_coupling()

        x0 = coupling.x0
        x1 = coupling.x1

        x0_idx = coord_to_idx(x0[:, 0], mins[0])
        y0_idx = coord_to_idx(x0[:, 1], mins[1])
        x1_idx = coord_to_idx(x1[:, 0], mins[0])
        y1_idx = coord_to_idx(x1[:, 1], mins[1])

        for i in range(len(x0_idx)):
            xs0, ys0 = x0_idx[i].item(), y0_idx[i].item()
            xs1, ys1 = x1_idx[i].item(), y1_idx[i].item()

            pts = bresenham(xs0, ys0, xs1, ys1)
            for (px, py) in pts:
                if 0 <= px < width and 0 <= py < height:
                    heatmap[py, px] += 1

    return heatmap
