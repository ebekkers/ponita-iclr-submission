import torch
import torch.nn as nn

from torch_geometric.nn import global_add_pool
from torch_geometric.nn import radius
from torch_geometric.utils import scatter

from siva.nn.layers import BundleConvNext, SeparableBundleConvNext
from siva.geometry.rotation import uniform_grid_s2, random_matrix
from siva.invariants import InvariantAttributes
from siva.nn.embedding.polynomial import PolynomialFeatures
from siva.utils.radial_cutoff import PolynomialCutoff
from torch_geometric.nn import global_mean_pool


class SIVA(nn.Module):
    """ Steerable E(3) equivariant (non-linear) convolutional network """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 n=12,
                 radius = 2.0,
                 M='R3S2',
                 basis_dim=None,
                 separable=True,
                 degree=2,
                 widening_factor=4):
        super().__init__()

        # ################ General graph settings
        self.radius = radius
        act_fn = torch.nn.GELU()

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
            self.fiber_basis_fn = nn.Sequential(nn.LazyLinear(hidden_dim), act_fn, nn.Linear(hidden_dim, basis_dim), act_fn)

        # ################ The main NN layers
        # Initial node embedding
        self.x_embedder = nn.Linear(input_dim, hidden_dim, bias=False)

        # The interaction layer
        if self.separable:
            InteractionLayer = SeparableBundleConvNext
        else:
            InteractionLayer = BundleConvNext
        
        # Make feedforward network
        self.interaction_layers = nn.ModuleList()
        self.read_out_layers = nn.ModuleList()
        for i in range(num_layers):
            self.interaction_layers.append(InteractionLayer(hidden_dim, basis_dim, act=act_fn, aggr="add", widening_factor=widening_factor))
            self.read_out_layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, graph):

        # ################ Unpack the Rd graph
        pos, x, batch = graph.pos, graph.x, graph.batch
        grid = self.grid

        # ################ Generate fiber grids
        if grid is not None:
            # Random grid (Give every graph its own grid)
            grid = self.grid.type_as(pos)
            # Randomly rotate the grid for each graph in the batch
            batch_size = batch.max() + 1
            rand_SO3 = random_matrix(batch_size).type_as(pos)  # [3, 3]
            grid = torch.einsum('bij,nj->bni', rand_SO3, grid)  # [batch_size, n, 3]

        # ################ Create the connectivity (edge_index)
        edge_index = radius(pos, pos, self.radius, batch_x = batch, batch_y = batch, max_num_neighbors=1000).flip(0)

        # ################ Precompute the pair-wise attributes and sample the basis functions (to be used in message passing)
        if grid is not None:
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
        basis = self.basis_fn(self.attr_embed_fn(edge_invariants))
        
        # ################ Perform the actual forward pass
        # Initial features
        x = self.x_embedder(x)  # Initial node embedding
        x = x[:, None, :].expand(-1, self.n, -1)  # Constant values along the fibers
        
        # Interaction layers
        local_pred = 0.  # Node-wise predictions after pooling over the fibers
        for interaction_layer, readout_layer in zip(self.interaction_layers, self.read_out_layers):
            # Interact
            if self.separable:
                x = interaction_layer(x, basis, fiber_basis, edge_index, batch)
            else:
                x = interaction_layer(x, basis, edge_index, batch)
            # Read out
            local_pred += readout_layer(x)
        
        # Make graph prediction via sum pooling
        pred = global_add_pool(local_pred.mean(dim=1), batch)

        return pred
