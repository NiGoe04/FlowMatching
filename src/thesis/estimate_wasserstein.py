import torch
from torch.utils.data import DataLoader

from src.experiments.mixed_gaussian_framework.scenarios import get_scenario
from src.flow_matching.model.coupling import Coupler
from src.flow_matching.model.distribution import GaussianMixtureDistribution
from src.flow_matching.model.losses import TensorCost
from src.flow_matching.view.utils import plot_tensor_2d, plot_tensor_3d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_transport_cost(x0, x1) -> torch.Tensor:
    return ((x0 - x1) ** 2).sum(dim=1).mean()

def get_distributions2(dist, dim):
    zeros = (dim - 2) * [0]
    means_x0 = [zeros + [-dist, -2], zeros + [-dist, 2]]
    means_x1 = [zeros + [dist, -2], zeros + [dist, 2]]
    variances = 0.1
    return GaussianMixtureDistribution(means_x0, variances, DEVICE), GaussianMixtureDistribution(means_x1, variances, DEVICE)

########################################################################################################################

SCENARIO = "tri_gauss_twice"
SIZE_TRAIN_SET = 2000
OT_BATCH_SIZE = 2000

def get_distributions(dim):
    return get_scenario(SCENARIO, dim, DEVICE)

dims = [2, 512]
#dims = [3, 4, 5, 6, 16, 64, 256, 1024]
plot_tensors = False

for dim in dims:
    gmd_x0, gmd_x1, _ = get_distributions(dim)
    x0_sample = gmd_x0.sample(SIZE_TRAIN_SET)
    x1_sample = gmd_x1.sample(SIZE_TRAIN_SET)
    coupler = Coupler(x0_sample, x1_sample)
    coupling = coupler.get_independent_coupling()

    loader = DataLoader(
        coupling,
        OT_BATCH_SIZE,
        shuffle=True,
    )

    tensor_list_x0 = []
    tensor_list_x1 = []

    for x_0, x_1 in loader:
        batch_size = x_1.shape[0]
        coupler_ot = Coupler(x_0, x_1)
        coupling_ot = coupler_ot.get_n_ot_coupling(batch_size, TensorCost.quadratic_cost)
        #coupling_ot = coupler_ot.get_independent_coupling()
        tensor_list_x0.append(coupling_ot.x0)
        tensor_list_x1.append(coupling_ot.x1)

    x0 = torch.cat(tensor_list_x0, dim=0)
    x1 = torch.cat(tensor_list_x1, dim=0)

    perm = torch.randperm(len(x0))
    x0_coupled = x0[perm]
    x1_coupled = x1[perm]

    w2_sq = compute_transport_cost(x0_coupled, x1_coupled)
    print("dim: {}, w2sq: {}".format(dim, w2_sq.item()))

    # plot tensors if required
    if plot_tensors and dim == 2:
        plot_tensor_2d(x0_sample)
        plot_tensor_2d(x1_sample)
    if plot_tensors and dim == 3:
        plot_tensor_3d(x0_sample)
        plot_tensor_3d(x1_sample)

'''
 dim: 3, w2sq: 16.345882415771484
dim: 16, w2sq: 17.74319839477539
dim: 64, w2sq: 25.32668685913086
dim: 256, w2sq: 59.77362823486328
dim: 1024, w2sq: 205.46359252929688
'''

'''
n = 100
dim: 3, w2sq: 16.700428009033203
dim: 4, w2sq: 16.04098129272461
dim: 5, w2sq: 16.764467239379883
dim: 6, w2sq: 16.50218391418457
dim: 16, w2sq: 18.555356979370117
dim: 64, w2sq: 25.738603591918945
dim: 256, w2sq: 60.26057434082031
dim: 1024, w2sq: 207.70257568359375
'''

'''
n = 100.000
dim: 3, w2sq: 16.33283805847168
dim: 4, w2sq: 16.388051986694336
dim: 5, w2sq: 16.505903244018555
dim: 6, w2sq: 16.567005157470703
dim: 16, w2sq: 17.71809196472168
dim: 64, w2sq: 25.31012725830078
dim: 256, w2sq: 59.73411178588867
dim: 1024, w2sq: 205.5005645751953
'''
