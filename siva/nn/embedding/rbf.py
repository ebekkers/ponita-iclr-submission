import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class RBF_euclidean(nn.Module):
    def __init__(self, domain_min, domain_max, n, codomain_dim):
        super().__init__()

        domain_dim = len(domain_min)

        grids = [torch.linspace(domain_min[i], domain_max[i], n[i]) for i in range(domain_dim)]
        self.register_buffer("grid", torch.stack(torch.meshgrid(grids), dim=-1).flatten(0, domain_dim-1))
        self.weight = torch.nn.parameter.Parameter(torch.randn([self.grid.shape[0], codomain_dim]))
        
        # Precompute the basis coefficients
        self.dist_fn = euclidean_distance
        self.rbf_width = self.dist_fn(self.grid[:, None], self.grid[None,:]).sort()[0][:, 1].mean()
        with torch.no_grad():
            m = rbf(self.dist_fn(self.grid[:, None, :], self.grid[None, :, :]), self.rbf_width)
            self.register_buffer("coeffs", torch.linalg.solve(m, self.weight))
       
    def forward(self, x):

        p = rbf(self.dist_fn(x[:, None, :], self.grid[None, :, :]), self.rbf_width)

        return p @ self.coeffs


class RBF_R3(nn.Module):
    def __init__(self, radius, n_x, codomain_dim):
        super().__init__()

        self.rbf_x = RBF_euclidean([0], [radius], [n_x], codomain_dim)

    def forward(self, edge_attr_relpos_r3):
        # edge_attr_relpos_r3 is a 1 dimensional vector (log invariant / the pair-wise distance)
        return self.rbf_x(edge_attr_relpos_r3)


class RBF_R3S2(nn.Module):
    def __init__(self, radius, n_x, n_R, codomain_dim):
        super().__init__()

        self.rbf_x = RBF_euclidean([0, -radius], [radius,radius], [n_x,2 * (n_x - 1) + 1], codomain_dim)
        self.rbf_R = RBF_euclidean([0.], [math.pi], [n_R], codomain_dim)

    def forward(self, edge_attr_relpos_r3s2):
        # edge_attr_relpos_r3s2 is a 3 dimensional vector (log invariants)
        # the first component c12 is the displacement in orthogonal direction to n and has range [0, radius]
        # the second component c3 is the discplacement along direction n and has range [-range, range]
        # the third component c45 is the geodesic distance on the sphere and has range [0, pi]
        c12c3 = edge_attr_relpos_r3s2[:,:2]
        c45 = edge_attr_relpos_r3s2[:,2:3]
        return self.rbf_x(c12c3) * self.rbf_R(c45)


class RBF_SE3(nn.Module):
    def __init__(self, radius, n_x, n_R, codomain_dim):
        super().__init__()

        self.rbf_x = RBF_euclidean([-radius]*3, [radius]*3, [n_x]*3, codomain_dim)
        self.rbf_R = RBF_euclidean([-math.pi]*3, [math.pi]*3, [n_R]*3, codomain_dim)

    def forward(self, edge_attr_relpos_se3):
        # edge_attr_relpos_se3 is a 6 dimensional vector (log invariants)
        # the first 3 components correspond to the spatial generators
        # the last 3 components to the rotation generators
        c123 = edge_attr_relpos_se3[:,:3]
        c456 = edge_attr_relpos_se3[:,3:]
        return self.rbf_x(c123) * self.rbf_R(c456)


def rbf(x, width):
    """
    2ln2 = 1.44269
    """
    return torch.exp(-(x**2 / (width**2 / 0.69314718)))


def euclidean_distance(x1, x2):
    return torch.linalg.norm(x1 - x2, dim=-1)