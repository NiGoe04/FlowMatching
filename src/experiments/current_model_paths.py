"""Auto-managed registry of trained model paths for mass experiments."""

# key format:
# "{dim}D|{ot_batch_size}V|{ot_solver_name}|{scenario_name}" for hungarian, vanilla fm mode
# "{dim}D|{ot_batch_size}V|{ot_solver_name}|eps={epsilon}|{scenario_name}" for sinkhorn, vanilla fm mode
# "{dim}D|{ot_batch_size}N|{ot_solver_name}|{scenario_name}" for hungarian, batch OT mode
# "{dim}D|{ot_batch_size}N|{ot_solver_name}|eps={epsilon}|{scenario_name}" for sinkhorn, batch OT mode
MODEL_PATHS: dict[str, str] = {
    '1024D|128N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_1024D_128N_hungarian_gaussian_circles_uftd_2026-02-27_16-08-23.pth',
    '1024D|16N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_1024D_16N_hungarian_gaussian_circles_uftd_2026-02-27_15-43-57.pth',
    '1024D|256N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_1024D_256N_hungarian_gaussian_circles_uftd_2026-02-27_14-45-26.pth',
    '1024D|256N|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_1024D_256N_hungarian_gaussian_mix_diff_var_1_2026-02-28_12-52-19.pth',
    '1024D|256V|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_1024D_256V_hungarian_gaussian_mix_diff_var_1_2026-02-28_13-06-57.pth',
    '1024D|32N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_1024D_32N_hungarian_gaussian_circles_uftd_2026-02-27_15-58-40.pth',
    '1024D|64N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_1024D_64N_hungarian_gaussian_circles_uftd_2026-02-27_16-04-08.pth',
    '128D|256N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_128D_256N_hungarian_gaussian_circles_uftd_2026-02-26_14-36-11.pth',
    '16D|256N|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_16D_256N_hungarian_gaussian_mix_diff_var_1_2026-02-28_13-54-17.pth',
    '16D|256V|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_16D_256V_hungarian_gaussian_mix_diff_var_1_2026-02-28_13-50-41.pth',
    '2048D|256N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_2048D_256N_hungarian_gaussian_circles_uftd_2026-02-27_14-49-58.pth',
    '2048D|256N|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_2048D_256N_hungarian_gaussian_mix_diff_var_1_2026-02-28_13-14-17.pth',
    '2048D|256V|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_2048D_256V_hungarian_gaussian_mix_diff_var_1_2026-02-28_13-09-33.pth',
    '256D|128N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_256D_128N_hungarian_gaussian_circles_uftd_2026-02-27_15-26-34.pth',
    '256D|16N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_256D_16N_hungarian_gaussian_circles_uftd_2026-02-27_15-10-15.pth',
    '256D|256N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_256D_256N_hungarian_gaussian_circles_uftd_2026-02-26_14-38-22.pth',
    '256D|256N|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_256D_256N_hungarian_gaussian_mix_diff_var_1_2026-02-28_12-48-47.pth',
    '256D|256V|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_256D_256V_hungarian_gaussian_mix_diff_var_1_2026-02-28_13-05-23.pth',
    '256D|32N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_256D_32N_hungarian_gaussian_circles_uftd_2026-02-27_15-17-47.pth',
    '256D|64N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_256D_64N_hungarian_gaussian_circles_uftd_2026-02-27_15-22-36.pth',
    '2D|256N|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_2D_256N_hungarian_gaussian_mix_diff_var_1_2026-02-28_13-30-39.pth',
    '2D|256V|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_2D_256V_hungarian_gaussian_mix_diff_var_1_2026-02-28_13-26-31.pth',
    '32D|256N|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_32D_256N_hungarian_gaussian_mix_diff_var_1_2026-02-28_13-40-56.pth',
    '32D|256V|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_32D_256V_hungarian_gaussian_mix_diff_var_1_2026-02-28_13-36-46.pth',
    '3D|256N|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_3D_256N_hungarian_gaussian_mix_diff_var_1_2026-02-28_13-35-30.pth',
    '3D|256V|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_3D_256V_hungarian_gaussian_mix_diff_var_1_2026-02-28_13-31-46.pth',
    '512D|256N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_512D_256N_hungarian_gaussian_circles_uftd_2026-02-27_14-39-38.pth',
    '64D|256N|hungarian|gaussian_circles_uftd': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_64D_256N_hungarian_gaussian_circles_uftd_2026-02-26_14-34-10.pth',
    '64D|256N|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_64D_256N_hungarian_gaussian_mix_diff_var_1_2026-02-28_12-46-24.pth',
    '64D|256V|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_64D_256V_hungarian_gaussian_mix_diff_var_1_2026-02-28_13-04-25.pth',
    '8D|256N|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_8D_256N_hungarian_gaussian_mix_diff_var_1_2026-02-28_13-50-05.pth',
    '8D|256V|hungarian|gaussian_mix_diff_var_1': 'C:\\Users\\niels\\PycharmProjects\\ML_Projects\\FlowMatching\\src\\experiments\\../../models\\model_8D_256V_hungarian_gaussian_mix_diff_var_1_2026-02-28_13-48-02.pth',
}
