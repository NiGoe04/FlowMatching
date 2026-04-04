from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from flow_matching.solver import ODESolver

from src.experiments.mixed_gaussian_framework.current_model_paths import MODEL_PATHS
from src.experiments.mixed_gaussian_framework.mass_training_mixed_gaussian import build_registry_key
from src.experiments.mixed_gaussian_framework.scenarios import SCENARIO_NAMES, get_scenario
from src.flow_matching.controller.utils import load_model_n_dim


def _solver_display_name(ot_optimizer: str, epsilon: Optional[float]) -> str:
    if ot_optimizer == "hungarian":
        return "hungarian"
    if ot_optimizer == "sinkhorn":
        if epsilon is None:
            raise ValueError("Sinkhorn optimizer requires --epsilon.")
        return f"sinkhorn_{epsilon}"
    raise ValueError(f"Unsupported ot optimizer: {ot_optimizer}")


def _resolve_model_path(
    *,
    framework_models_dir: Path,
    dim: int,
    ot_batch_size: int,
    ot_optimizer: str,
    epsilon: Optional[float],
    scenario_name: str,
    vanilla_fm_mode: bool,
) -> Path:
    key = build_registry_key(dim, ot_batch_size, ot_optimizer, scenario_name, epsilon, vanilla_fm_mode)
    candidate = MODEL_PATHS.get(key)

    if candidate is not None:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

        # Registry can contain machine-specific absolute paths. Try a local fallback with basename.
        basename_fallback = framework_models_dir / candidate_path.name
        if basename_fallback.exists():
            return basename_fallback

    solver_name = _solver_display_name(ot_optimizer, epsilon)
    mode_char = "V" if vanilla_fm_mode else "N"
    pattern = f"model_{dim}D_{ot_batch_size}{mode_char}_{solver_name}_{scenario_name}_*.pth"
    matches = sorted(framework_models_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "No matching model found. "
        f"Searched registry key '{key}' and pattern '{pattern}' in '{framework_models_dir}'."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained mixed-gaussian model, sample x0 from the chosen scenario, "
            "push samples through the ODE solver, and print generated target samples."
        )
    )
    parser.add_argument("--dim", type=int, required=True, help="Dimension d used during training.")
    parser.add_argument("--ot-batch-size", type=int, required=True, help="OT batch size k used during training.")
    parser.add_argument("--scenario", required=True, choices=SCENARIO_NAMES, help="Scenario name.")
    parser.add_argument(
        "--ot-optimizer",
        default="hungarian",
        choices=["hungarian", "sinkhorn"],
        help="OT optimizer used in training.",
    )
    parser.add_argument("--epsilon", type=float, default=None, help="Sinkhorn epsilon (required when --ot-optimizer sinkhorn).")
    parser.add_argument(
        "--vanilla-fm-mode",
        action="store_true",
        help="Use registry key for vanilla FM mode (V). Default is non-vanilla/mixed (N).",
    )
    parser.add_argument("--amount-samples", type=int, default=16, help="How many samples to generate.")
    parser.add_argument("--solver-method", default="euler", help="ODE solver method.")
    parser.add_argument("--solver-steps", type=int, default=100, help="Number of solver steps.")
    parser.add_argument("--t-end", type=float, default=1.0, help="End time for sampling trajectory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.ot_optimizer == "sinkhorn" and args.epsilon is None:
        raise ValueError("--epsilon must be provided when --ot-optimizer sinkhorn.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    framework_dir = Path(__file__).resolve().parents[1] / "experiments" / "mixed_gaussian_framework"
    models_dir = framework_dir / "models"

    model_path = _resolve_model_path(
        framework_models_dir=models_dir,
        dim=args.dim,
        ot_batch_size=args.ot_batch_size,
        ot_optimizer=args.ot_optimizer,
        epsilon=args.epsilon,
        scenario_name=args.scenario,
        vanilla_fm_mode=args.vanilla_fm_mode,
    )

    gmd_x0, gmd_x1, _ = get_scenario(scenario_name=args.scenario, dim=args.dim, device=device)
    x0_samples = gmd_x0.sample(args.amount_samples)

    model = load_model_n_dim(args.dim, str(model_path), device=device)
    solver = ODESolver(velocity_model=model)
    generated_target_samples = solver.sample(
        x_init=x0_samples,
        method=args.solver_method,
        step_size=1.0 / float(args.solver_steps),
        time_grid=torch.tensor([0.0, float(args.t_end)], device=device),
    )

    true_target_samples = gmd_x1.sample(args.amount_samples)

    print(f"Using device: {device}")
    print(f"Using model: {model_path}")
    print(f"Scenario: {args.scenario} | dim={args.dim} | ot_batch_size={args.ot_batch_size}")
    print("\nSource samples x0:")
    print(x0_samples)
    print("\nGenerated target samples (model output):")
    print(generated_target_samples)
    print("\nReference target samples (fresh draw from target distribution x1):")
    print(true_target_samples)


if __name__ == "__main__":
    main()
