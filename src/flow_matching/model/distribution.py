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
        # Gaussian noise with mean 0 and specified variance
        std = variance ** 0.5
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

    def merge(self, other: 'Distribution'):
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