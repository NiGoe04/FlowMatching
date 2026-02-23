"""Auto-managed registry of trained model paths for mass experiments."""

# key format:
# "{dim}D|{ot_batch_size}N|{ot_solver_name}|{scenario_name}" for hungarian
# "{dim}D|{ot_batch_size}N|{ot_solver_name}|eps={epsilon}|{scenario_name}" for sinkhorn
MODEL_PATHS: dict[str, str] = {}
