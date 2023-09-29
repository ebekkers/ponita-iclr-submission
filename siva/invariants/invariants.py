import torch
from siva.invariants.r3s2 import invariant_attributes_r3s2
from siva.invariants.rd import invariant_attributes_rd


class InvariantAttributes(torch.nn.Module):
    def __init__(self, manifold, method): 
        super().__init__()
        self.manifold = manifold
        self.method = method

    def forward(self, pos_s, pos_t=None, grid_s=None, grid_t=None, separable=False):
        # Shape of outputs
        # If grid_mode:
        #   if seperable:
        #       [num_edges, num_grid_t, 2], [num_grid_t, num_grid_s, 1]
        #   else:
        #       [num_edges, num_grid, num_grid, 3]
        # else:
        #   [num_edges, 3]
        
        # Check dimension
        dim_pos = pos_s.shape[-1]
        dim_grid = grid_s.shape[-1] if grid_s is not None else 0

        # Call corresponding embedding function
        if self.manifold == "R3":
            if dim_pos == 3:
                return invariant_attributes_rd(pos_s, pos_t)
            else:
                raise ValueError(f"For \"R3\" the input pos should be of dimension 3")
        if self.manifold == "R3S2":
            if (dim_pos == 6) or (dim_pos == 3 and dim_grid == 3):
                return invariant_attributes_r3s2(pos_s, pos_t, grid_s, grid_t, separable, method=self.method)
            else:
                raise ValueError(f"For \"R3S2\" the input pos should 6-dimensional or 3-dimensional if also a 3-dimensional grid is provided")
        if self.manifold == "SE3":
            if (dim_pos == 12) or (dim_pos == 3 and dim_grid == 9):
                raise NotImplementedError("R3xSO3 invariants not implemented yet...")
            else:
                raise ValueError(f"For \"SE3\" the input pos should 12-dimensional or 3-dimensional if also a 9-dimensional grid is provided")