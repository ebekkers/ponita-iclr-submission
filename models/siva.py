import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from siva.nn import RFF, AggregationLayer
from siva.nn import ConvNextLayer as InteractionLayer
from siva.graph.transforms import RdToM
from siva.nn.embedding.rbf import RBF_R3, RBF_R3S2, RBF_SE3

from siva.graph.lift import LiftRdToM, ComputeEdgeAttr
from torch_geometric.nn import radius

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
                 M='R3S2'):
        super().__init__()

        # Attribute dimensions
        edge_attr_dim, node_attr_dim, dim_x, dim_R, self.pairwise = manifold_dims(M)
        self.dim_x, self.dim_R = dim_x, dim_R
        self.node_attr_Q = node_attr_dim != 0
        edge_attr_dim += 2 * node_attr_dim  # Because we also concat the node invariants (if present)
        if self.pairwise:
            input_dim *= 2  # We concatenate feature on both end of the edge in the Rd graph

        # Graph construction layers (lifting, edge attr fn, projection layer)
        self.lifting_fn = LiftRdToM(M, n, lifting_radius)
        self.edge_attr_fn = ComputeEdgeAttr(M, lie_algebra=True)
        self.projection_fn = AggregationLayer('mean')

        # Other graph settings
        self.radius = radius
        self.task = task
        act_fn = torch.nn.SiLU()

        # Initial feature embedding
        self.x_embedder = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                             act_fn,
                                             nn.Linear(hidden_dim, hidden_dim))
        # Edge attribute embedding
        self.edge_attr_embedder_x = nn.Sequential(
                RFF(dim_x, hidden_dim, sigma=[sigma_x]*dim_x),
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, hidden_dim),
                act_fn
            )
        self.edge_attr_embedder_R = nn.Sequential(
                RFF(dim_R, hidden_dim, sigma=[sigma_R]*dim_R),
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, hidden_dim),
                act_fn
            )
        
        # Message passing layers
        layers = []
        post_interaction_layers = []
        for i in range(num_layers):
            layers.append(InteractionLayer(hidden_dim, hidden_dim, dropout=droprate, act=act_fn, aggr="add", layer_norm=layer_norm))
            post_interaction_layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            act_fn,
                                            nn.Linear(hidden_dim, output_dim)))
        self.layers = nn.ModuleList(layers)
        self.post_interaction_layers = nn.ModuleList(post_interaction_layers)


    def forward(self, graph):

        # Unpack the Rd graph
        pos_Rd = graph.pos
        x_Rd = graph.x
        batch_Rd = graph.batch
        # And construct the graph on M
        pos_M, x_M, batch_M, edge_index_proj = self.lifting_fn(pos_Rd, x_Rd, batch_Rd)

        # Construct edge_index and attr for lifting convolution (use flip to have "from" (index 0) -> "to" (index 1)
        edge_index_Rd_to_M = radius(pos_Rd, pos_M[:,:3], self.radius, batch_x = batch_Rd, batch_y = batch_M, max_num_neighbors=1000).flip(0)
        edge_attr_Rd_to_M = self.edge_attr_fn(pos_Rd[edge_index_Rd_to_M[0]], pos_M[edge_index_Rd_to_M[1]])
        # Construct kernel (window with radial envelope)
        edge_dist_Rd_to_M = torch.linalg.norm(pos_Rd[edge_index_Rd_to_M[0],:3]-pos_M[edge_index_Rd_to_M[1],:3], dim=-1, keepdim=True)
        edge_window_Rd_to_M = dist_window(edge_dist_Rd_to_M, self.radius)
        kernel_basis_Rd_to_M = self.edge_attr_embedder_x(edge_attr_Rd_to_M) * edge_window_Rd_to_M
        
        # Construct edge_index and attr (embedding) for the group convolutions 
        edge_index_M_to_M = radius(pos_M[:,:3], pos_M[:,:3], self.radius, batch_x = batch_M, batch_y = batch_M, max_num_neighbors=1000).flip(0)
        edge_attr_M_to_M = self.edge_attr_fn(pos_M[edge_index_M_to_M[0]], pos_M[edge_index_M_to_M[1]])
        # Construct kernel (window with radial envelope)
        edge_dist_M_to_M = torch.linalg.norm(pos_M[edge_index_M_to_M[0],:3]-pos_M[edge_index_M_to_M[1],:3], dim=-1, keepdim=True)
        edge_window_M_to_M = dist_window(edge_dist_M_to_M, self.radius)
        kernel_basis_M_to_M = self.edge_attr_embedder_x(edge_attr_M_to_M[:,:self.dim_x]) * self.edge_attr_embedder_R(edge_attr_M_to_M[:,self.dim_x:]) * edge_window_M_to_M

        # Initial features
        x = self.x_embedder(x_Rd)

        # Interaction layers
        lifting_layer = True
        x_Rd_pred = 0.
        for layer, post_interaction_layer in zip(self.layers, self.post_interaction_layers):
            if lifting_layer:
                x = layer(x, kernel_basis_Rd_to_M, edge_index_Rd_to_M, batch_M, batch_Rd)
                lifting_layer = False
            else:
                x = layer(x, kernel_basis_M_to_M, edge_index_M_to_M, batch_M)
            x_Rd = self.projection_fn(x, edge_index_proj)
            x_Rd_pred +=  post_interaction_layer(x_Rd)
        pred = global_add_pool(x_Rd_pred, batch_Rd)

        return pred

def dist_window(dist, max_dist, min_dist=0.):
    range = max_dist - min_dist
    return 0.5 * (torch.cos(torch.pi * (dist - min_dist) / range) + 1)

# edge_attr_dim, node_attr_dim, attr_dim_x, attr_dim_R, self.pairwise 
def manifold_dims(M):
    if M == 'R2':
        return 2, 0, 1, 0, False
    if M == 'SE2':
        return 3, 0, 2, 1, False
    if M == 'R3':
        return 1, 0, 1, 0, False
    if M == 'R3S2':
        return 3, 0, 2, 1, False
    if M == 'SE3':
        return 6, 0, 3, 3, False



