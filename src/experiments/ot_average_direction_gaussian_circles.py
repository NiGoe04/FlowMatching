from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datetime import datetime

from src.experiments.scenarios import get_scenario
from src.flow_matching.model.coupling import Coupler
from src.flow_matching.model.losses import TensorCost


def _draw_batch_excluding_index(pool: torch.Tensor, n: int, excluded_idx: int) -> torch.Tensor:
    if n < 2:
        raise ValueError("n must be > 1.")
    if len(pool) < n:
        raise ValueError(f"Pool size ({len(pool)}) must be >= n ({n}).")

    device = pool.device
    all_indices = torch.arange(len(pool), device=device)
    valid_indices = all_indices[all_indices != excluded_idx]

    selection = valid_indices[torch.randperm(len(valid_indices), device=device)[: n - 1]]
    return pool[selection]


def _plot_result(
    sampled_x0: torch.Tensor,
    sampled_x1: torch.Tensor,
    fixed_x0: torch.Tensor,
    x1_prime: torch.Tensor,
    output_path: Path | None,
    epochs: int,
    n: int,
    sample_size: int,
    save_plot: bool,
) -> Path | None:
    """Plot both sampled distributions and highlight fixed_x0 / x1'."""

    x0_np = sampled_x0.detach().cpu().numpy()
    x1_np = sampled_x1.detach().cpu().numpy()
    fixed_x0_np = fixed_x0.detach().cpu().numpy()
    x1_prime_np = x1_prime.detach().cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.scatter(x0_np[:, 0], x0_np[:, 1], s=10, alpha=0.35, color="royalblue", label="sampled x0")
    plt.scatter(x1_np[:, 0], x1_np[:, 1], s=10, alpha=0.35, color="darkorange", label="sampled x1")

    plt.scatter(
        fixed_x0_np[0],
        fixed_x0_np[1],
        s=140,
        color="crimson",
        edgecolors="black",
        linewidths=1.2,
        marker="o",
        label="fixed x0",
        zorder=5,
    )

    plt.scatter(
        x1_prime_np[0],
        x1_prime_np[1],
        s=160,
        color="limegreen",
        edgecolors="black",
        linewidths=1.2,
        marker="*",
        label="x1' (x0 + avg direction)",
        zorder=6,
    )

    plt.plot(
        [fixed_x0_np[0], x1_prime_np[0]],
        [fixed_x0_np[1], x1_prime_np[1]],
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
    )

    plt.title(
        f"gaussian_circles: OT average direction from fixed x0\n"
        f"number samples = {sample_size}, epochs = {epochs}, n = {n}"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()

    if save_plot:
        if output_path is None:
            raise ValueError("output_path must be provided if save_plot=True.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=180)
        plt.close()
        return output_path
    else:
        plt.show()
        plt.close()
        return None


def run(
    simulated_epochs: int = 200,
    n: int = 32,
    sample_size: int = 2000,
    seed: int = 42,
    plot_output_path: str | Path | None = None,
    save_plot: bool = True,
) -> dict[str, torch.Tensor | str | None]:

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gmd_x0, gmd_x1, _ = get_scenario("gaussian_circles", dim=2, device=device)

    sampled_x0 = gmd_x0.sample(sample_size)
    sampled_x1 = gmd_x1.sample(sample_size)

    torch.manual_seed(torch.seed())

    fixed_x0_idx = 0 # torch.randint(0, sample_size - 1, ()).item()
    fixed_x0 = sampled_x0[fixed_x0_idx]

    remembered_vectors = []

    for _ in range(simulated_epochs):
        x1_batch_idx = torch.randperm(len(sampled_x1), device=device)[:n]
        x1_batch = sampled_x1[x1_batch_idx]

        x0_rest = _draw_batch_excluding_index(sampled_x0, n=n, excluded_idx=fixed_x0_idx)
        x0_batch = torch.cat([fixed_x0.unsqueeze(0), x0_rest], dim=0)

        coupling = Coupler(x0_batch, x1_batch).get_n_ot_coupling(
            n=n, cost_fn=TensorCost.quadratic_cost
        )

        fixed_pair_idx = torch.argmin(
            torch.linalg.norm(coupling.x0 - fixed_x0.unsqueeze(0), dim=1)
        )
        optimal_x1 = coupling.x1[fixed_pair_idx]

        remembered_vectors.append(optimal_x1 - fixed_x0)

    remembered_vectors_tensor = torch.stack(remembered_vectors, dim=0)
    average_direction = remembered_vectors_tensor.mean(dim=0)
    x1_prime = fixed_x0 + average_direction

    plot_path = _plot_result(
        sampled_x0=sampled_x0,
        sampled_x1=sampled_x1,
        fixed_x0=fixed_x0,
        x1_prime=x1_prime,
        output_path=Path(plot_output_path) if plot_output_path else None,
        epochs=simulated_epochs,
        n=n,
        sample_size=sample_size,
        save_plot=save_plot,
    )

    return {
        "fixed_x0": fixed_x0,
        "average_direction": average_direction,
        "x1_prime": x1_prime,
        "all_vectors": remembered_vectors_tensor,
        "plot_path": str(plot_path) if plot_path else None,
    }


if __name__ == "__main__":

    SAVE_PLOT = False

    seed = 45 #random.randint(0, 10000)

    base_path = "plots/ot_average_direction_gaussian_circles"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path_with_timestamp = f"{base_path}_{timestamp}.png"

    result = run(
        simulated_epochs=1000,
        n=16,
        sample_size=1000,
        seed=seed,
        plot_output_path=path_with_timestamp,
        save_plot=SAVE_PLOT,
    )

    print("Fixed x0:", result["fixed_x0"].detach().cpu().numpy())
    print("Average direction:", result["average_direction"].detach().cpu().numpy())
    print("x1' = x0 + average_direction:", result["x1_prime"].detach().cpu().numpy())
    print("Saved plot:", result["plot_path"])