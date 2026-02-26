"""Auto-managed registry of trained model paths for mass experiments."""

# key format:
# "{dim}D|{ot_batch_size}V|{ot_solver_name}|{scenario_name}" for hungarian, vanilla fm mode
# "{dim}D|{ot_batch_size}V|{ot_solver_name}|eps={epsilon}|{scenario_name}" for sinkhorn, vanilla fm mode
# "{dim}D|{ot_batch_size}N|{ot_solver_name}|{scenario_name}" for hungarian, batch OT mode
# "{dim}D|{ot_batch_size}N|{ot_solver_name}|eps={epsilon}|{scenario_name}" for sinkhorn, batch OT mode
MODEL_PATHS: dict[str, str] = {
    '128D|256N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_128D_256N_hungarian_gaussian_circles_uftd_2026-02-26_14-36-11.pth',
    '256D|256N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_256D_256N_hungarian_gaussian_circles_uftd_2026-02-26_14-38-22.pth',
    '64D|256N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_64D_256N_hungarian_gaussian_circles_uftd_2026-02-26_14-34-10.pth',
}
