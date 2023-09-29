import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool, BatchNorm
from torch_geometric.nn.pool import fps
from siva.nn import RFF, AggregationLayer
from siva.nn import ConvNextLayerSEd as InteractionLayer
from siva.graph.transforms import RdToM
from siva.nn.embedding.rbf import RBF_R3, RBF_R3S2, RBF_SE3

from siva.graph.lift import LiftRdToM, ComputeEdgeAttr
from torch_geometric.nn import radius, knn
from torch_geometric.utils import scatter

from siva.geometry.r3s2 import random_rotation
from siva.geometry.rotation import random_matrix, uniform_grid, uniform_grid_s2

class SIVA(nn.Module):
    """ Steerable E(3) equivariant (non-linear) convolutional network """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 droprate=0.0,
                 task='graph',
                 lifting_radius=None,
                 n=12,
                 radius = 2.0,
                 sigma_x=1.,
                 sigma_R=1.,
                 min_dist=0.,
                 layer_norm=False,
                 M='R3S2',
                 grid_mode=True,
                 basis_dim=None,
                 depthwise=True):
        super().__init__()

        # Attribute dimensions
        self.dim_x, self.dim_R = manifold_dims(M)
        if basis_dim is None:
            basis_dim = hidden_dim

        # Graph construction layers (lifting, edge attr fn, projection layer)
        self.grid_mode = grid_mode
        self.lifting_fn = LiftRdToM(M, n, lifting_radius, permute_grid=True, grid_mode=grid_mode)
        self.grid = self.lifting_fn.grid
        self.edge_attr_fn = ComputeEdgeAttr(M, lie_algebra=True)
        self.projection_fn = AggregationLayer('mean')

        # Other graph settings
        # self.radius = radius
        self.radius = 0.005
        self.task = task
        act_fn = torch.nn.GELU()

        # Initial feature embedding
        self.x_embedder = nn.Sequential(
            nn.Linear(input_dim+3, hidden_dim, bias=True),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim, bias=True)
        )

        # Edge attribute embedding
        self.basis_fn_x = nn.Sequential(
                RFF(self.dim_x, hidden_dim, sigma=[sigma_x]*self.dim_x),
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, basis_dim),
                act_fn
            )
        self.basis_fn_R = nn.Sequential(
                RFF(self.dim_R, hidden_dim, sigma=[sigma_R]*self.dim_R),
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, basis_dim),
                act_fn
            )
        
        # Message passing layers
        layers = []
        down_layers = []
        feature_dim = hidden_dim
        for i in range(num_layers):
            down_layers.append(DownsampleBlock(feature_dim, 2 * feature_dim, down_sample = 4, k=16))
            feature_dim = feature_dim * 2
            layers.append(ConvolverBlock(feature_dim, basis_dim, k=16, act_fn = act_fn, depthwise=depthwise))
        self.layers = nn.ModuleList(layers)
        self.down_layers = nn.ModuleList(down_layers)

        self.post_interaction_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            act_fn,
            nn.Linear(256, 64),
            act_fn,
            nn.Linear(64, output_dim))
        

    def forward(self, graph):

        # Unpack the Rd graph
        pos_Rd = graph.pos.type(torch.get_default_dtype())
        x_Rd = torch.ones_like(pos_Rd[:,0])[:,None]  # no features
        batch_Rd = graph.batch
        # And construct the graph on M
        pos_M, x_M, batch_M, edge_index_proj = self.lifting_fn(pos_Rd, x_Rd, batch_Rd)
        grid = self.lifting_fn.grid
        x_M = torch.einsum('mi,bi->bm', grid, graph.normal.type(torch.get_default_dtype()))[:,:,None]  # Project normals to the grid
        # x_M = torch.concat([x_M, pos_M[:,:,:3], graph.normal[:,None,:].expand(-1,grid.shape[0],-1)], dim=-1)
        x_M = torch.concat([x_M, pos_M[:,:,:3]], dim=-1)

        # Initial features
        x = self.x_embedder(x_M)

        # Interaction layers
        for down_layer, layer in zip(self.down_layers, self.layers):
            x, pos_Rd, batch_Rd = down_layer(x, pos_Rd, batch_Rd)
            x = layer(x, pos_Rd, batch_Rd, grid, self.basis_fn_x, self.basis_fn_R)

        # Point-wise classification
        x = global_mean_pool(torch.mean(x, dim=1), batch_Rd)
        pred = self.post_interaction_layer(x)

        return pred


class ConvolverBlock(torch.nn.Module):
    def __init__(self, in_dim, basis_dim, k, act_fn = torch.nn.GELU(), depthwise=False): 
        super().__init__()

        # Edge attribute embedding
        self.basis_fn_x = nn.Sequential(
                RFF(2, in_dim, sigma=[10,10]),
                nn.Linear(in_dim, in_dim),
                act_fn,
                nn.Linear(in_dim, basis_dim),
                act_fn
            )
        self.basis_fn_R = nn.Sequential(
                RFF(1, in_dim, sigma=[10]),
                nn.Linear(in_dim, in_dim),
                act_fn,
                nn.Linear(in_dim, basis_dim),
                act_fn
            )

        self.interaction_layer = InteractionLayer(in_dim, basis_dim, act=act_fn, aggr="mean", depthwise=depthwise)
        self.interaction_layer2 = InteractionLayer(in_dim, basis_dim, act=act_fn, aggr="mean", depthwise=depthwise)
        self.k = k


    def forward(self, x, pos_Rd, batch_Rd, grid, basis_fn_x, basis_fn_R):

        edge_index = knn(pos_Rd, pos_Rd, k=self.k, batch_x = batch_Rd, batch_y = batch_Rd).flip(0)
        rel_pos_x, rel_pos_R, edge_dist = edge_attr_r3s2_grid_mode(pos_Rd, grid, edge_index)
        kernel_basis_x = self.basis_fn_x(rel_pos_x)
        kernel_basis_R = self.basis_fn_R(rel_pos_R)

        x = self.interaction_layer(x, kernel_basis_x, kernel_basis_R, edge_index, batch_Rd) # todo: remove batch option
        x = self.interaction_layer2(x, kernel_basis_x, kernel_basis_R, edge_index, batch_Rd) # todo: remove batch option

        return x
    
class DownsampleBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, down_sample, k): 
        super().__init__()

        self.down_sample = down_sample
        self.k = k
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = BatchNorm(out_dim)
        self.act_fn = torch.nn.ReLU()


    def forward(self, x_Rd, pos_Rd, batch_Rd):
        index = fps(pos_Rd, batch_Rd, ratio=1/self.down_sample)
        edge_index = knn(pos_Rd, pos_Rd[index], k=self.k, batch_x = batch_Rd, batch_y = batch_Rd[index]).flip(0)

        # Downsample: linear + batch norm + relu + max pool
        x = self.linear(x_Rd)
        # x = self.norm(x.flatten(0,1)).unflatten(0, (x.shape[0], x.shape[1]))
        x = self.act_fn(x)
        x = scatter(x[edge_index[0]], edge_index[1], dim=0, reduce='max')

        return x, pos_Rd[index], batch_Rd[index]

def dist_window(dist, max_dist, min_dist=0.):
    range = max_dist - min_dist
    return 0.5 * (torch.cos(torch.pi * (dist - min_dist) / range) + 1)

# edge_attr_dim, node_attr_dim, attr_dim_x, attr_dim_R, self.pairwise 
def manifold_dims(M):
    if M == 'R2':
        return 1, 0
    if M == 'SE2':
        return 2, 1
    if M == 'R3':
        return 1, 0
    if M == 'R3S2':
        return 2, 1
    if M == 'SE3':
        return 3, 3


def edge_attr_r3s2_grid_mode(pos_Rd, grid, edge_index):
    rel_pos_R = torch.einsum('mi,ni->mn', grid, grid)[:,:,None]
    rel_pos = pos_Rd[edge_index[0]] - pos_Rd[edge_index[1]]
    rel_pos_z = torch.einsum('bi,ni->bn', rel_pos, grid)  # component in z-direction
    rel_pos_xy = torch.linalg.norm(rel_pos[:,None,:] - rel_pos_z[:,:,None] * grid[None,:,:], dim=-1)
    rel_pos_x = torch.stack([rel_pos_xy, rel_pos_z], dim=-1)
    edge_dist = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
    return rel_pos_x, rel_pos_R, edge_dist
