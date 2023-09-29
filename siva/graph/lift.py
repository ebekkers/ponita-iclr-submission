import torch
from torch_geometric.nn import radius_graph
from siva.geometry.se3 import r3_edge_to_se3, edge_attr_se3
from siva.geometry.r3s2 import r3_edge_to_r3s2, edge_attr_r3s2
from siva.geometry.rotation import random_matrix, uniform_grid, uniform_grid_s2


class LiftRdToM(torch.nn.Module):
    def __init__(self, M, n = None, radius = None, permute_grid = True, grid_mode = False): 
        super().__init__()

        self.permute_grid = permute_grid
        self.grid_mode = grid_mode
        self.n = n

        # Set manifold
        if M in ["R3S2", "SE3"]:
            self.M = M
        else:
            raise ValueError(f"Unknown manifold M (should be R3S2 or SE3)")

        # Initialize grid or set radius
        if not ((n is None) ^ (radius is None)):  # xor
            raise ValueError(f"Either n or radius should be specified, but not both or neither (so xor)")
        elif n is not None:
            self.radius = None
            if self.M == 'SE3':
                # self.grid = uniform_grid(n, "matrix")
                self.register_buffer('grid', uniform_grid(n, "matrix"))
            elif self.M == 'R3S2':
                # self.grid = uniform_grid_s2(n)
                self.register_buffer('grid', uniform_grid_s2(n))
        elif radius is not None:
            # self.grid = None
            self.register_buffer('grid', None)
            self.radius = radius
        
        # The lifting function
        if self.M == "R3S2":
            self.lifting_fn = lift_R3_to_R3S2
        elif self.M == "SE3":
            self.lifting_fn = lift_R3_to_SE3

    def forward(self, pos, x, batch):
        # Random permute the grid
        if self.permute_grid:
            rand_SO3 = random_matrix(1)[0].type_as(self.grid)  # [3, 3]
            if self.M == "R3S2":
                self.grid = torch.einsum('ij,nj->ni', rand_SO3, self.grid)  # [n, 3]
            elif self.M == "SE3":
                self.grid = torch.einsum('ij,njk->nik', rand_SO3, self.grid)  # [n, 3, 3]
        self.grid = self.grid.type_as(pos)
        # Do the lifting
        pos_M, x_M, batch_M, edge_index_proj = self.lifting_fn(pos, x, batch, self.grid, self.radius)
        # If in grid_mode 
        if self.grid_mode:
            pos_M = pos_M.unflatten(0, (-1, self.n))
            x_M = x_M.unflatten(0, (-1, self.n))
            batch_M = batch_M.unflatten(0, (-1, self.n))
        # Return the graph
        return pos_M, x_M, batch_M, edge_index_proj


class ComputeEdgeAttr(torch.nn.Module):
    def __init__(self, M, lie_algebra): 
        super().__init__()

        # Set manifold
        if M in ["R3S2", "SE3"]:
            self.M = M
        else:
            raise ValueError(f"Unknown manifold M1 or M2 (should be R3S2 or SE3)")
        
        if self.M == "R3S2":
            self.attr_fn = edge_attr_r3s2
        if self.M == "SE3":
            self.attr_fn = edge_attr_se3
        
        self.lie_algebra = lie_algebra

    def forward(self, pos_s, pos_t):
        return self.attr_fn(pos_s, pos_t, self.lie_algebra)


def lift_R3_to_R3S2(pos, x, batch, grid = None, radius = None):
    # Either a radius or a number of group elements in SO(3) is specified
    # When a radius is specified, the graph is lifted via the edge_index of a radius graph
    # When a grid is specified, this grid (of SO(3)) is assigned to each spatial location.

    # Create the lifted graph nodes (pos_M, x_M, batch_M)
    if (grid is not None) and (radius is not None):
        raise ValueError(f"Either grid or radius should be specified, but not both")
    elif radius is not None:
        # Construct edges to use as reference directions
        edge_index_for_lifting = radius_graph(pos, radius, batch, loop=False, max_num_neighbors=1000)
        # Use edges to define a position-orientation graph
        pos_M, x_M, batch_M = r3_edge_to_r3s2(pos, x, batch, edge_index_for_lifting)
        # Projection edge index
        node_idx_Rd = torch.arange(0, pos.shape[0]).type_as(batch)
        node_idx_M = torch.arange(0, pos_M.shape[0]).type_as(batch)
        edge_index_proj = torch.stack([node_idx_M, node_idx_Rd[edge_index_for_lifting[1]]], dim=0)
    elif grid is not None:
        b = pos.shape[0]
        n = grid.shape[0]
        pos_x = pos[:, None, :].repeat(1, n, 1).flatten(0,1)  # [b*n, 3]
        # Assign rotation matrix to each base point, and random rotate the grids at each base point b
        pos_S2 = grid[None, :, :].repeat(b, 1, 1).flatten(0,1)  # [b*n, 3]
        # Concatenate to create Rd x S2 elements
        pos_M = torch.cat([pos_x, pos_S2], dim=-1)  # [b*n, 6]
        # x_M and batch_M are just the copied values at the base points
        x_M = x[:, None, :].repeat(1, n, 1).flatten(0,1)  # [b*n, c]
        batch_M = batch[:, None].repeat(1, n).flatten(0,1)  # [b*n, ]
        # edge_index for projecting back to the base points
        edge_index_proj = torch.stack([torch.arange(0, b*n), torch.arange(0, b)[:, None].repeat(1, n).flatten(0,1)], dim=0).type_as(batch)
    else:
        raise ValueError(f"Neither grid nor radius is specified")

    return pos_M, x_M, batch_M, edge_index_proj


def lift_R3_to_SE3(pos, x, batch, grid = None, radius = None):
    # Either a radius or a number of group elements in SO(3) is specified
    # When a radius is specified, the graph is lifted via the edge_index of a radius graph
    # When a grid is specified, this grid (of SO(3)) is assigned to each spatial location.

    # Create the lifted graph nodes (pos_M, x_M, batch_M)
    if (grid is not None) and (radius is not None):
        raise ValueError(f"Either grid or radius should be specified, but not both")
    elif radius is not None:
        # Construct edges to use as reference directions
        edge_index_for_lifting = radius_graph(pos, radius, batch, loop=False, max_num_neighbors=1000)
        # Use edges to define a position-orientation graph
        pos_M, x_M, batch_M = r3_edge_to_se3(pos, x, batch, edge_index_for_lifting, radius)
        # Projection edge index
        node_idx_Rd = torch.arange(0, pos.shape[0]).type_as(batch)
        node_idx_M = torch.arange(0, pos_M.shape[0]).type_as(batch)
        edge_index_proj = torch.stack([node_idx_M, node_idx_Rd[edge_index_for_lifting[1]]], dim=0)
    elif grid is not None:
        b = pos.shape[0]
        n = grid.shape[0]
        pos_x = pos[:, None, :].repeat(1, n, 1).flatten(0,1)  # [b*n, 3]
        # Assign rotation matrix to each base point, and random rotate the grids at each base point b
        pos_SO3 = grid[None, :, :, :].repeat(b, 1, 1, 1).flatten(0,1)  # [b*n, 3, 3]
        # Concatenate to create Rd x SO3 elements (rotation matrices flattened)
        pos_M = torch.cat([pos_x, pos_SO3.flatten(-2,-1)], dim=-1)  # [b*n, 12]
        # x_M and batch_M are just the copied values at the base points
        x_M = x[:, None, :].repeat(1, n, 1).flatten(0,1)  # [b*n, c]
        batch_M = batch[:, None].repeat(1, n).flatten(0,1)  # [b*n, ]
        # edge_index for projecting back to the base points
        edge_index_proj = torch.stack([torch.arange(0, b*n), torch.arange(0, b)[:, None].repeat(1, n).flatten(0,1)], dim=0).type_as(batch)
    else:
        raise ValueError(f"Neither grid nor radius is specified")

    return pos_M, x_M, batch_M, edge_index_proj