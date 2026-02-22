import torch

from src.flow_matching.model.distribution import GaussianMixtureDistribution

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

variances = 0.1
num_samples = 1000
DIMS = [2, 3, 5, 8, 21, 89, 144, 233]

# -------------------------
# Two-component mixture
# -------------------------
print("\n==============================")
print("Two-component Gaussian mixture")
print("==============================\n")

for d in DIMS:
    if d < 2:
        raise ValueError("d must be >= 2")

    # [0, ..., 0, 2, -2] and [0, ..., 0, 2, 2]
    prefix = [0] * (d - 2)
    means = [
        prefix + [2, -2],
        prefix + [2,  2],
    ]

    dist = GaussianMixtureDistribution(means, variances, device=DEVICE)
    samples = dist.sample(num_samples)

    nll = dist.nll_mi_corrected(samples)
    nll_per_dim = dist.nll_mi_corrected_per_dim(samples)

    print(f"========== d = {d} ==========")
    print(f"NLL: {nll}")
    print(f"NLL per dim: {nll_per_dim}")
    print()


# -------------------------
# Single Gaussian (one center)
# -------------------------
print("\n==============================")
print("Single isotropic Gaussian")
print("==============================\n")

for d in DIMS:
    # choose a single mean (center) of the same dimension
    # (here: all zeros; you can change this if you want)
    mean = [0] * d

    dist = GaussianMixtureDistribution([mean], variances, device=DEVICE)
    samples = dist.sample(num_samples)

    nll = dist.nll_mi_corrected(samples)
    nll_per_dim = dist.nll_mi_corrected_per_dim(samples)

    print(f"========== d = {d} ==========")
    print(f"NLL: {nll}")
    print(f"NLL per dim: {nll_per_dim}")
    print()