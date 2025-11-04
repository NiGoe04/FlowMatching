import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class LRFinder:
    """
    A simple learning rate finder for PyTorch models.
    Sweeps learning rates exponentially and tracks loss.
    """
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.lrs = []
        self.losses = []

    def range_test(self, loader: DataLoader, lr_start=1e-6, lr_end=1.0, num_iters=100):
        """
        Perform the LR range test.
        """
        self.model.train()
        # Store original state
        orig_state = self.model.state_dict()
        orig_optimizer_state = self.optimizer.state_dict()

        lr_mult = (lr_end / lr_start) ** (1 / num_iters)
        lr = lr_start
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        iteration = 0
        for x_0, x_1 in loader:
            iteration += 1
            if iteration > num_iters:
                break

            x_0 = x_0.to(self.device)
            x_1 = x_1.to(self.device)

            t = torch.rand(x_0.size(0), device=self.device)
            # your path.sample call here if needed, e.g.
            # sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            # x_t, dx_t = sample.x_t, sample.dx_t
            x_t, dx_t = x_1, x_0  # placeholder: replace with your actual sample

            self.optimizer.zero_grad()
            output = self.model(x_t, t)
            loss = self.criterion(output, dx_t)
            loss.backward()
            self.optimizer.step()

            self.lrs.append(lr)
            self.losses.append(loss.item())

            # Update LR
            lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # Restore model and optimizer state
        self.model.load_state_dict(orig_state)
        self.optimizer.load_state_dict(orig_optimizer_state)

    def plot(self, skip_start=10, skip_end=5):
        """
        Plot loss vs learning rate.
        """
        lrs = self.lrs[skip_start:-skip_end]
        losses = self.losses[skip_start:-skip_end]

        plt.figure(figsize=(8, 5))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel("Learning rate (log scale)")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.show()
