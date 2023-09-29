from torch import nn
import torch
import math
from torch.nn import LayerNorm
from siva.geometry.rotation import uniform_grid_s2, random_matrix
from siva.invariants import InvariantAttributes
from siva.nn.embedding.polynomial import PolynomialFeatures
from siva.utils.radial_cutoff import PolynomialCutoff
from torch_geometric.nn import radius
from siva.nn.layers import BundleConvNext
from siva.nn.layers import SeparableBundleConvNext
from torch_geometric.nn import global_mean_pool


class SIVA(nn.Module):
    """ Steerable E(3) equivariant (non-linear) convolutional network """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 input_dim_vec = 0,
                 output_dim_vec = 0,
                 n=12,
                 radius = 100.0,
                 M='R3S2',
                 basis_dim=None,
                 separable=True,
                 degree=2, widening_factor=4, device='gpu'):
        super().__init__()

        # ################ General graph settings
        self.radius = radius
        act_fn = torch.nn.GELU()
        self.device = device

        # ################ Geometric setup
        if M=="R3":
            self.fiber_bundle = False
            self.n = 1
            self.grid = None
            self.separable = False
        elif M=="R3S2":
            self.fiber_bundle = True
            self.n = n
            self.grid = uniform_grid_s2(n)  # [n,d]
            self.separable = separable

        # ################ Pair-wise attributes and embedding functions
        # For computing pair-wise attributes
        self.invariant_attr_fn = InvariantAttributes(M, "Euclidean")

        # Attribute embedding functions
        basis_dim = hidden_dim if (basis_dim is None) else basis_dim
        self.attr_embed_fn = PolynomialFeatures(degree)
        self.basis_fn = nn.Sequential(nn.LazyLinear(hidden_dim), act_fn, nn.Linear(hidden_dim, basis_dim), act_fn)
        if self.separable:
            self.fiber_attr_embed_fn = PolynomialFeatures(degree)
            self.fiber_basis_fn = nn.Sequential(nn.Linear(degree, hidden_dim), act_fn, nn.Linear(hidden_dim, basis_dim), act_fn)

        # ################ The main NN layers
        # Initial node embedding
        self.x_embedder = nn.Linear(input_dim + input_dim_vec, hidden_dim, bias=False)

        # The interaction layer
        if self.separable:
            InteractionLayer = SeparableBundleConvNext
        else:
            InteractionLayer = BundleConvNext
        
        # Make feedforward network
        self.interaction_layers = nn.ModuleList()
        for i in range(num_layers):
            self.interaction_layers.append(InteractionLayer(hidden_dim, basis_dim, act=act_fn, aggr="add", widening_factor=4, layer_scale=None))
        self.readout_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), act_fn, nn.Linear(hidden_dim, output_dim + output_dim_vec))

    def forward(self, graph):

        # ################ Unpack the Rd graph
        pos, batch, vel, charges, edge_index = graph.pos, graph.batch, graph.vel, graph.charges, graph.edge_index
        grid = self.grid

        # ################ Generate fiber grids
        if grid is not None:
            # Random grid (Give every graph its own grid)
            grid = self.grid.type_as(pos)
            # Randomly rotate the grid for each graph in the batch
            batch_size = batch.max() + 1
            rand_SO3 = random_matrix(batch_size).type_as(pos)  # [3, 3]\
            grid = torch.einsum('bij,nj->bni', rand_SO3, grid)  # [batch_size, n, 3]
        
        # ################ Precompute the pair-wise attributes and sample the basis functions (to be used in message passing)
        if grid is not None:
            # Copying the grid to each node
            grid_node = grid[batch]
            # Copying the grid to each edge
            grid = grid[batch[edge_index[0]]]  # [num_nodes, n, 3]
        # Compute the invariant edge attributes and the within fiber pair-wise invariants
        #   Note, if separable it returns a tuple (edge_invariants,fiber_invariants) with 
        #   edge_invariants.shape = [num_edges, num_grid_points, num_inv] and fiber_invariants.shape = [num_grid_points, num_grid_points, num_inv]
        #   If not separable it returns only edge_invariants with shape [num_edges, num_grid_points, num_grid_points, num_invariants]
        edge_invariants = self.invariant_attr_fn(pos[edge_index[0]], pos[edge_index[1]], grid, separable=self.separable)  # If separable it returns a tuple, otherwise only edge_attr
        if grid is None:
            edge_invariants = edge_invariants[:, None, None, :]  # from shape [num_edges, num_inv] to [num_edges, 1, 1, num_inv]
        
        # Sample the basis functions
        if self.separable:
            # Unpack spatial/fiber invariants and sample fiber basis
            edge_invariants, fiber_invariants = edge_invariants
            fiber_basis = self.fiber_basis_fn(self.fiber_attr_embed_fn(fiber_invariants))
        
        # Sample and window spatial basis
        edge_invariants = torch.cat([edge_invariants,(charges[edge_index[0]]*charges[edge_index[1]])[:,None].expand(-1, self.n, -1)],-1)
        basis = self.basis_fn(self.attr_embed_fn(edge_invariants))
        
        # ################ Perform the actual forward pass
        # Initial features
        vel_on_sphere = self.vec2sphere(vel, grid_node)
        rel_pos = pos - global_mean_pool(pos, batch)[batch]
        rel_pos_on_sphere = self.vec2sphere(rel_pos, grid_node)
        charge_on_sphere = self.scalar2sphere(charges)
        vel_norm = vel.norm(dim=-1, keepdim=True)
        vel_norm_on_sphere = self.scalar2sphere(vel_norm)

        # Initial feature embeding
        x = torch.cat([vel_on_sphere, rel_pos_on_sphere, charge_on_sphere, vel_norm_on_sphere], dim=-1)
        x = self.x_embedder(x)
        
        # Interaction layers
        for interaction_layer in self.interaction_layers:
            # Interact
            if self.separable:
                x = interaction_layer(x, basis, fiber_basis, edge_index, batch)
            else:
                x = interaction_layer(x, basis, edge_index, batch)
        
        # Output
        v = self.sphere2vec(self.readout_layer(x),grid_node)

        return v

    def vec2sphere(self, vec, grid):
        return torch.einsum('bd,bnd->bn', vec, grid)[:,:,None]
    
    def scalar2sphere(self, scalar):
        return scalar[:, None, :].expand(-1, self.n, -1)
    
    def sphere2vec(self, spherical_signal, grid):
        return torch.mean(grid * spherical_signal, dim = -2)