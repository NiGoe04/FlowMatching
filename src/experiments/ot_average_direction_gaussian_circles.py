from __future__ import annotations

import torch

from src.experiments.scenarios import get_scenario
from src.flow_matching.model.coupling import Coupler
from src.flow_matching.model.losses import TensorCost


def _draw_batch_excluding_index(pool: torch.Tensor, n: int, excluded_idx: int) -> torch.Tensor:
    """Draw n-1 random points from pool excluding a specific index."""
    if n < 2:
        raise ValueError("n must be > 1.")
    if len(pool) < n:
        raise ValueError(f"Pool size ({len(pool)}) must be >= n ({n}).")

    device = pool.device
    all_indices = torch.arange(len(pool), device=device)
    valid_indices = all_indices[all_indices != excluded_idx]

    if len(valid_indices) < n - 1:
        raise ValueError("Not enough samples left after excluding fixed x0 index.")

    selection = valid_indices[torch.randperm(len(valid_indices), device=device)[: n - 1]]
    return pool[selection]


def run(simulated_epochs: int = 200, n: int = 32, sample_size: int = 1000, seed: int = 42) -> dict[str, torch.Tensor]:
    """
    Simulate repeated OT couplings to estimate an average transport direction from a fixed x0.

    Steps:
      1) Build gaussian_circles distributions (2D), sample fixed subsets.
      2) Select one fixed x0 from the inner circle sample.
      3) For each epoch, draw a fresh x1 subset of size n and an x0 batch that includes fixed x0.
      4) Compute OT coupling on the batch and keep the matched x1 for fixed x0.
      5) Average matched directions and compute x1' = x0 + avg_direction.
    """
    if simulated_epochs <= 0:
        raise ValueError("simulated_epochs must be > 0.")
    if n <= 1:
        raise ValueError("n must be > 1.")

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gmd_x0, gmd_x1, _ = get_scenario("gaussian_circles", dim=2, device=device)

    sampled_x0 = gmd_x0.sample(sample_size)
    sampled_x1 = gmd_x1.sample(sample_size)

    # Pick a particular x0 from the inner circle sample: smallest radius in sampled_x0.
    radii = torch.linalg.norm(sampled_x0, dim=1)
    fixed_x0_idx = int(torch.argmin(radii).item())
    fixed_x0 = sampled_x0[fixed_x0_idx]

    remembered_vectors = []

    for _ in range(simulated_epochs):
        x1_batch_idx = torch.randperm(len(sampled_x1), device=device)[:n]
        x1_batch = sampled_x1[x1_batch_idx]

        x0_rest = _draw_batch_excluding_index(sampled_x0, n=n, excluded_idx=fixed_x0_idx)
        x0_batch = torch.cat([fixed_x0.unsqueeze(0), x0_rest], dim=0)

        coupling = Coupler(x0_batch, x1_batch).get_n_ot_coupling(n=n, cost_fn=TensorCost.quadratic_cost)

        # Locate the pair for the fixed x0 (first element in x0_batch).
        fixed_pair_idx = torch.argmin(torch.linalg.norm(coupling.x0 - fixed_x0.unsqueeze(0), dim=1))
        optimal_x1 = coupling.x1[fixed_pair_idx]

        remembered_vectors.append(optimal_x1 - fixed_x0)

    remembered_vectors_tensor = torch.stack(remembered_vectors, dim=0)
    average_direction = remembered_vectors_tensor.mean(dim=0)
    x1_prime = fixed_x0 + average_direction

    return {
        "fixed_x0": fixed_x0,
        "average_direction": average_direction,
        "x1_prime": x1_prime,
        "all_vectors": remembered_vectors_tensor,
    }


if __name__ == "__main__":
    result = run(simulated_epochs=200, n=32, sample_size=1000, seed=42)

    print("Fixed x0:", result["fixed_x0"].detach().cpu().numpy())
    print("Average direction:", result["average_direction"].detach().cpu().numpy())
    print("x1' = x0 + average_direction:", result["x1_prime"].detach().cpu().numpy())
