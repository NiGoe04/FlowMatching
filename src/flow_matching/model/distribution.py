import torch

GENERIC_CENTER = [1, 1]
GENERIC_SAMPLES = 1

class Distribution:
    def __init__(self, center, n_samples, device):
        super().__init__()
        center_tensor = torch.tensor(center, dtype=torch.float32)
        self.device = device
        self.tensor = center_tensor.unsqueeze(0).repeat(n_samples, 1).to(device)

    @classmethod
    def get_any(cls, device):
        return cls(GENERIC_CENTER, GENERIC_SAMPLES, device)

    def set_to(self, distribution_tensor):
        self.tensor = distribution_tensor.to(self.device)
        return self

    def with_uniform_noise(self, noise_bound=0.05):
        # uniform noise in [-noise_bound, +noise_bound]
        uniform_noise = (torch.rand_like(self.tensor) * 2 - 1) * noise_bound
        self.tensor += uniform_noise
        return self

    def with_gaussian_noise(self, variance=0.05):
        # Gaussian noise with mean 0 and specified variance.
        # Variance can be scalar, list, or tensor broadcastable to self.tensor.
        variance_tensor = torch.as_tensor(variance, dtype=self.tensor.dtype, device=self.tensor.device)
        if torch.any(variance_tensor < 0):
            raise ValueError("Variance must be non-negative.")
        std = torch.sqrt(variance_tensor)
        gaussian_noise = torch.randn_like(self.tensor) * std
        self.tensor += gaussian_noise
        return self

    def with_bounded_gaussian_noise(self, variance=0.05, bound_radius=0.1):
        std = variance ** 0.5
        gaussian_noise = torch.randn_like(self.tensor) * std  # [N, D]
        # Compute L2 norm of each vector (row-wise)
        norms = torch.norm(gaussian_noise, dim=-1, keepdim=True)  # [N, 1]
        # Scale down vectors whose norm exceeds bound_radius
        scale = torch.clamp(norms, max=bound_radius) / (norms + 1e-8)
        bounded_noise = gaussian_noise * scale  # scale vectors into circle
        self.tensor += bounded_noise
        return self

    def shifted_by(self, shift_vector):
        shift_tensor = torch.tensor(shift_vector, dtype=self.tensor.dtype, device=self.tensor.device)
        self.tensor += shift_tensor
        return self

    def merged_with(self, other: 'Distribution'):
        if not isinstance(other, Distribution):
            raise ValueError("Can only merge with another Distribution object.")
        self.tensor = torch.cat([self.tensor, other.tensor], dim=0)
        return self

class Distribution2D(Distribution):
    def __init__(self, center, n_samples, device):
        super().__init__(center, n_samples, device)
        assert len(center) == 2

    def with_uniform_noise_x(self, noise_bound=0.05):
        noise = (torch.rand(self.tensor.shape[0], 1, device=self.tensor.device) * 2 - 1) * noise_bound
        self.tensor[:, 0:1] += noise
        return self

    def with_uniform_noise_y(self, noise_bound=0.05):
        noise = (torch.rand(self.tensor.shape[0], 1, device=self.tensor.device) * 2 - 1) * noise_bound
        self.tensor[:, 1:2] += noise
        return self

    def with_gaussian_noise_x(self, variance=0.05):
        std = variance ** 0.5
        noise = torch.randn(self.tensor.shape[0], 1, device=self.tensor.device) * std
        self.tensor[:, 0:1] += noise
        return self

    def with_gaussian_noise_y(self, variance=0.05):
        std = variance ** 0.5
        noise = torch.randn(self.tensor.shape[0], 1, device=self.tensor.device) * std
        self.tensor[:, 1:2] += noise
        return self


class GaussianMixtureDistribution:
    def __init__(self, means, variances, device):
        self.device = device
        self.means = torch.as_tensor(means, dtype=torch.float32, device=device)
        if self.means.dim() != 2:
            raise ValueError("means must have shape [n_components, d].")

        if isinstance(variances, (float, int)):
            if variances < 0:
                raise ValueError("Variance must be non-negative.")
            self.variances = torch.full_like(self.means, float(variances))
        else:
            self.variances = torch.as_tensor(variances, dtype=torch.float32, device=device)
            if self.variances.shape != self.means.shape:
                raise ValueError("variances must have shape [n_components, d].")
            if torch.any(self.variances < 0):
                raise ValueError("Variance must be non-negative.")

        self.n_components, self.d = self.means.shape

    def sample(self, n):
        counts = torch.multinomial(
            torch.ones(self.n_components, device=self.device),
            n,
            replacement=True,
        ).bincount(minlength=self.n_components)

        merged_distribution = None
        for component_idx, component_count in enumerate(counts.tolist()):
            if component_count == 0:
                continue

            component_distribution = Distribution(
                center=self.means[component_idx].tolist(),
                n_samples=component_count,
                device=self.device,
            ).with_gaussian_noise(variance=self.variances[component_idx])

            if merged_distribution is None:
                merged_distribution = component_distribution
            else:
                merged_distribution = merged_distribution.merged_with(component_distribution)

        if merged_distribution is None:
            return torch.empty((0, self.d), dtype=torch.float32, device=self.device)

        permutation = torch.randperm(merged_distribution.tensor.shape[0], device=self.device)
        return merged_distribution.tensor[permutation]

    def _component_log_probs(self, x1):
        eps = torch.finfo(x1.dtype).eps
        safe_variances = torch.clamp(self.variances, min=eps)

        diff = x1.unsqueeze(1) - self.means.unsqueeze(0)
        scaled_sq = (diff ** 2) / safe_variances.unsqueeze(0)
        quadratic = scaled_sq.sum(dim=-1)

        log_det = torch.log(2 * torch.pi * safe_variances).sum(dim=-1)
        component_log_probs = -0.5 * (quadratic + log_det.unsqueeze(0))
        return component_log_probs

    def nll(self, x1):
        x1 = torch.as_tensor(x1, dtype=torch.float32, device=self.device)
        if x1.dim() != 2 or x1.shape[1] != self.d:
            raise ValueError(f"x1 must have shape [n, {self.d}].")

        component_log_probs = self._component_log_probs(x1)

        log_mixture_prob = torch.logsumexp(component_log_probs, dim=1) - torch.log(
            torch.tensor(self.n_components, dtype=x1.dtype, device=self.device)
        )
        return -log_mixture_prob.mean()

    def nll_per_dim(self, x1):
        return self.nll(x1) / self.d

    def nll_mi_corrected(self, x1):
        """
         Mutual-information corrected NLL.

         NLL_corrected = NLL - I(X;Z)
         where I(X;Z) = log K - H(Z|X)
         """
        x1 = torch.as_tensor(x1, dtype=torch.float32, device=self.device)
        nll = self.nll(x1)
        component_log_probs = self._component_log_probs(x1)

        log_post = component_log_probs - torch.logsumexp(
            component_log_probs, dim=1, keepdim=True
        )

        post = torch.exp(log_post)

        # 3) Conditional entropy H(Z|X)
        H_Z_given_X = -(post * log_post).sum(dim=1).mean()

        # 4) Mutual information I(X;Z)
        logK = torch.log(
            torch.tensor(self.n_components, dtype=x1.dtype, device=self.device)
        )
        I_XZ = logK - H_Z_given_X

        # 5) Corrected NLL
        nll_corrected = nll - I_XZ

        return nll_corrected

    def nll_mi_corrected_per_dim(self, x1):
        return self.nll_mi_corrected(x1) / self.d