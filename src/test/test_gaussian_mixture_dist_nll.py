from src.flow_matching.model.distribution import GaussianMixtureDistribution

import sys
import torch

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

torch.set_default_dtype(DTYPE)

# Make randomness reproducible
GLOBAL_SEED = 1234
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

# Test sizes (increase if you want tighter estimates)
N_SAMPLES_STABILITY = 50_000
N_SAMPLES_MONO = 50_000

# Tolerances (Monte Carlo estimates won't be perfect)
TOL_STABILITY_PER_DIM = 0.03      # allowed spread across d for corrected NLL/d
TOL_MONO_EPS = 1e-3               # monotonic check slack

# -----------------------------
# Helper assertions
# -----------------------------
def assert_close(a, b, tol, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg} | abs({a} - {b}) = {abs(a-b)} > {tol}")

def assert_monotone_increasing(vals, eps=0.0, msg=""):
    # Checks vals[i+1] >= vals[i] - eps
    for i in range(len(vals) - 1):
        if vals[i + 1] < vals[i] - eps:
            raise AssertionError(
                f"{msg} | Not monotone at i={i}: {vals[i+1]} < {vals[i]} (eps={eps})"
            )

def pretty(x):
    return float(x) if isinstance(x, (float, int)) else float(x.item())

# -----------------------------
# Scenario builders
# -----------------------------
def build_iid_two_component_gmm(GaussianMixtureDistribution, d, sep=2.0, var=1.0, device=DEVICE):
    """
    Creates a 2-component diagonal GMM in R^d where each dimension is i.i.d.
    Component 0 mean = 0
    Component 1 mean = sep  (in every dimension)
    Variance = var (in every dimension, both components)

    This is "the same distribution" across different d in the sense:
    each coordinate has the same 1D mixture distribution, independent across dims.
    """
    means = torch.stack([
        torch.zeros(d, device=device),
        torch.full((d,), sep, device=device),
    ], dim=0).to(torch.float32)

    variances = torch.full((2, d), var, device=device, dtype=torch.float32)

    return GaussianMixtureDistribution(means=means, variances=variances, device=device)

def build_shifted_gmm(GaussianMixtureDistribution, base_gmm, shift, device=DEVICE):
    """
    Shift all component means by a constant vector shift in every dimension.
    """
    K, d = base_gmm.means.shape
    shift_vec = torch.full((d,), shift, device=device, dtype=torch.float32)
    means = base_gmm.means + shift_vec.unsqueeze(0)  # [K,d]
    variances = base_gmm.variances.clone()
    return GaussianMixtureDistribution(means=means, variances=variances, device=device)

def build_variance_scaled_gmm(GaussianMixtureDistribution, base_gmm, var_scale, device=DEVICE):
    """
    Multiply all variances by var_scale (keeps means same).
    """
    means = base_gmm.means.clone()
    variances = base_gmm.variances * float(var_scale)
    return GaussianMixtureDistribution(means=means, variances=variances, device=device)

# -----------------------------
# (Optional) If your class doesn't yet support return_details, add this to it:
#
# def nll_mi_corrected_per_dim(self, x1, return_details: bool = False):
#     out = self.nll_mi_corrected(x1, return_details=return_details)
#     if return_details:
#         out["nll_per_dim"] = out["nll"] / self.d
#         out["nll_corrected_per_dim"] = out["nll_corrected"] / self.d
#         out["I_XZ_per_dim"] = out["I_XZ"] / self.d
#         return out
#     return out / self.d
# -----------------------------

# -----------------------------
# TEST 1:
# MI-corrected NLL per dim should be consistent across dimensions
# for truly i.i.d. per-dimension distribution
# -----------------------------
def test_corrected_nll_per_dim_stability(GaussianMixtureDistribution):
    print("=" * 80)
    print("TEST 1) MI-corrected NLL per dim stability across dimensions")
    print(f"Device: {DEVICE}")
    print("-" * 80)

    dims = [2, 5, 20, 100, 250]
    sep = 2.0
    var = 1.0

    corrected_vals = []
    baseline_vals = []
    mi_vals = []
    hz_vals = []

    for d in dims:
        gmm = build_iid_two_component_gmm(GaussianMixtureDistribution, d=d, sep=sep, var=var)

        # Sample from the model itself (so we measure "self NLL")
        x = gmm.sample(N_SAMPLES_STABILITY)

        # Baseline NLL/d (typically drifts with d for mixtures)
        baseline = pretty(gmm.nll_per_dim(x))

        # Corrected NLL/d + diagnostics
        stats = gmm.nll_mi_corrected_per_dim(x, return_details=True)
        corrected = pretty(stats["nll_corrected_per_dim"])
        I_XZ = pretty(stats["I_XZ"])
        H_Z_given_X = pretty(stats["H_Z_given_X"])

        corrected_vals.append(corrected)
        baseline_vals.append(baseline)
        mi_vals.append(I_XZ)
        hz_vals.append(H_Z_given_X)

        print(f"d={d:>4} | baseline NLL/d={baseline:.6f} | corrected NLL/d={corrected:.6f} "
              f"| I(X;Z)={I_XZ:.6f} | H(Z|X)={H_Z_given_X:.6f}")

    spread = max(corrected_vals) - min(corrected_vals)
    print("-" * 80)
    print(f"Corrected NLL/d range across dims: min={min(corrected_vals):.6f}, "
          f"max={max(corrected_vals):.6f}, spread={spread:.6f}")

    if spread > TOL_STABILITY_PER_DIM:
        raise AssertionError(
            f"Corrected NLL/d is not stable enough across dims: spread={spread:.6f} "
            f"> tol={TOL_STABILITY_PER_DIM}"
        )

    print("✅ PASS: corrected NLL/d is stable across dimensions (within tolerance).")

# -----------------------------
# TEST 2:
# NLL should increase as the distribution moves further away
# We'll test multiple mismatch types:
#   (A) mean shift grows -> NLL grows monotonically
#   (B) variance scale mismatch grows -> NLL grows away from optimum at scale=1
# -----------------------------
def test_nll_increases_with_distance(GaussianMixtureDistribution):
    print("=" * 80)
    print("TEST 2) NLL increases with distribution mismatch / distance")
    print(f"Device: {DEVICE}")
    print("-" * 80)

    d = 50
    sep = 2.0
    var = 1.0
    base = build_iid_two_component_gmm(GaussianMixtureDistribution, d=d, sep=sep, var=var)

    # Draw samples from the base distribution
    x = base.sample(N_SAMPLES_MONO)

    # ---- 2A) Mean shift monotonicity ----
    shifts = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]
    baseline_nll_d = []
    corrected_nll_d = []

    print("2A) Mean-shift mismatch:")
    for s in shifts:
        model = build_shifted_gmm(GaussianMixtureDistribution, base, shift=s)
        baseline = pretty(model.nll_per_dim(x))
        corr = model.nll_mi_corrected_per_dim(x)  # scalar
        corr = pretty(corr)
        baseline_nll_d.append(baseline)
        corrected_nll_d.append(corr)
        print(f"  shift={s:>4.2f} | baseline NLL/d={baseline:.6f} | corrected NLL/d={corr:.6f}")

    # Both should be monotone increasing with shift (allow tiny eps due to MC noise)
    assert_monotone_increasing(baseline_nll_d, eps=TOL_MONO_EPS,
                               msg="Baseline NLL/d should increase with mean shift")
    assert_monotone_increasing(corrected_nll_d, eps=TOL_MONO_EPS,
                               msg="Corrected NLL/d should increase with mean shift")

    print("✅ PASS: NLL/d increases monotonically with mean shift (baseline and corrected).")

    # ---- 2B) Variance mismatch ----
    # Here the minimum should be near var_scale=1, and it should increase as you move away.
    var_scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    nlls = []
    print("-" * 80)
    print("2B) Variance-scale mismatch (minimum near scale=1.0):")
    for vs in var_scales:
        model = build_variance_scaled_gmm(GaussianMixtureDistribution, base, var_scale=vs)
        val = pretty(model.nll_per_dim(x))
        nlls.append(val)
        print(f"  var_scale={vs:>4.2f} | baseline NLL/d={val:.6f}")

    # Check that scale=1.0 is (approximately) the minimum among tested points
    min_idx = min(range(len(nlls)), key=lambda i: nlls[i])
    if var_scales[min_idx] != 1.0:
        raise AssertionError(
            f"Expected minimum near var_scale=1.0 but got minimum at var_scale={var_scales[min_idx]} "
            f"(values={nlls})"
        )

    print("✅ PASS: NLL/d is minimized near the true variance scale (1.0).")

# -----------------------------
# Extra: Optional stress test to show why baseline NLL/d drifts but corrected is stable
# -----------------------------
def demo_baseline_vs_corrected_drift(GaussianMixtureDistribution):
    print("=" * 80)
    print("DEMO) Baseline drift vs corrected stability (quick illustration)")
    print("-" * 80)

    dims = [2, 5, 20, 100, 250]
    sep = 2.0
    var = 1.0

    for d in dims:
        gmm = build_iid_two_component_gmm(GaussianMixtureDistribution, d=d, sep=sep, var=var)
        x = gmm.sample(20_000)

        baseline = pretty(gmm.nll_per_dim(x))
        stats = gmm.nll_mi_corrected_per_dim(x, return_details=True)
        corrected = pretty(stats["nll_corrected_per_dim"])
        I_per_dim = pretty(stats["I_XZ_per_dim"])

        print(f"d={d:>4} | baseline NLL/d={baseline:.6f} | corrected NLL/d={corrected:.6f} "
              f"| I/d={I_per_dim:.6f}")

# -----------------------------
# Main runner
# -----------------------------
def main():
    # Import your implementation here.
    # Adjust the import to wherever your class lives.
    #
    # Example:
    # from my_module import GaussianMixtureDistribution
    #
    # For now, we expect it to already be in scope.
    try:
        GaussianMixtureDistribution  # noqa: F821
    except NameError:
        print(
            "ERROR: GaussianMixtureDistribution is not in scope.\n"
            "Please import it at the top of this file (see comments in main())."
        )
        sys.exit(1)

    test_corrected_nll_per_dim_stability(GaussianMixtureDistribution)
    test_nll_increases_with_distance(GaussianMixtureDistribution)
    demo_baseline_vs_corrected_drift(GaussianMixtureDistribution)

    print("=" * 80)
    print("ALL TESTS PASSED ✅")

if __name__ == "__main__":
    main()