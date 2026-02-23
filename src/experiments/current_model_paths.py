"""Auto-managed registry of trained model paths for mass experiments."""

# key format:
# "{dim}D|{ot_batch_size}N|{ot_solver_name}|{scenario_name}" for hungarian
# "{dim}D|{ot_batch_size}N|{ot_solver_name}|eps={epsilon}|{scenario_name}" for sinkhorn
MODEL_PATHS: dict[str, str] = {
    '233D|256N|hungarian|double_gauss_twice': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_233D_256N_hungarian_double_gauss_twice_2026-02-23_13-57-33.pth',
    '233D|256N|hungarian|double_gauss_twice_ftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_233D_256N_hungarian_double_gauss_twice_ftd_2026-02-23_14-15-41.pth',
    '377D|256N|hungarian|double_gauss_twice': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_377D_256N_hungarian_double_gauss_twice_2026-02-23_14-12-12.pth',
    '377D|256N|hungarian|double_gauss_twice_ftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_377D_256N_hungarian_double_gauss_twice_ftd_2026-02-23_14-16-50.pth',
    '610D|256N|hungarian|double_gauss_twice': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_610D_256N_hungarian_double_gauss_twice_2026-02-23_14-13-23.pth',
    '610D|256N|hungarian|double_gauss_twice_ftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_610D_256N_hungarian_double_gauss_twice_ftd_2026-02-23_14-17-57.pth',
    '987D|256N|hungarian|double_gauss_twice': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_987D_256N_hungarian_double_gauss_twice_2026-02-23_14-14-33.pth',
    '987D|256N|hungarian|double_gauss_twice_ftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_987D_256N_hungarian_double_gauss_twice_ftd_2026-02-23_14-19-17.pth',
}
