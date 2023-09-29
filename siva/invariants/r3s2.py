import torch


def invariant_attributes_r3s2(pos_s, pos_t, grid_s=None, grid_t=None, separable=False, method="Euclidean", eps=1e-12):
    # Shape of outputs
    # If grid_mode:
    #   if seperable:
    #       [num_edges, num_grid_t, 2], [num_grid_t, num_grid_s, 1]
    #   else:
    #       [num_edges, num_grid, num_grid, 3]
    # else:
    #   [num_edges, 3]
    if method=="Euclidean":
        if grid_s is None:
            return invariant_attributes_r3s2_euclidean(pos_s, pos_t)
        else:
            return invariant_attributes_r3s2_euclidean_grid(pos_s, pos_t, grid_s, grid_t=grid_t, separable=separable, eps=eps)
    elif method=="Euler":
        if grid_s is None:
            return invariant_attributes_r3s2_euler(pos_s, pos_t, eps=eps)
        else:
            return invariant_attributes_r3s2_euler_grid(pos_s, pos_t, grid_s, grid_t=grid_t, separable=separable, eps=eps)





def invariant_attributes_r3s2_euclidean(pos_s, pos_t):
    # Spatial positions and local orientations
    x_s, n_s, x_t, n_t = pos_s[:, :3], pos_s[:, 3:], pos_t[:, :3], pos_t[:, 3:]

    # Decompose relative position in translation along n_t and orthogonal to it
    rel_pos = x_t - x_s
    invariant_1 = (rel_pos * n_t).sum(dim=-1, keepdim=True)                 # rel_pos_component_along_n_t
    invariant_2 = (rel_pos - invariant_1 * n_t).norm(dim=-1, keepdim=True)  # rel_pos_component_orthogonal_to_n_t
    
    # Relative orientation as innerproduct between orientations ( note: angle = arccos(invariant_3) )
    invariant_3 = (n_s * n_t).sum(dim=-1, keepdim=True)                     # rel_ori_component_along_n_t

    # Concatenate and return (shape = [num_pos, 3])
    return torch.cat([invariant_1, invariant_2, invariant_3], dim=-1)


def invariant_attributes_r3s2_euclidean_grid(pos_s, pos_t, grid_s, grid_t=None, separable=False, eps=1e-12):
    # Spatial positions
    x_s, x_t = pos_s[..., :3], pos_t[..., :3]
    num_pos = pos_s.shape[0]
    # Orientations
    n_s = grid_s
    n_t = n_s if grid_t is None else grid_t
    # grid sizes
    num_n_s, num_n_t = n_s.shape[-2], n_t.shape[-2]
    # Check if each data point has their own grid
    grid_dim = n_s.dim()
    invariant_1_einsum_eq = 'bi,ni->bn' if (grid_dim == 2) else 'bi,bni->bn'
   
    # Decompose relative position in polar coordinates (relative to n_t), shape  of each invariant will be [num_pos, num_t, 1]
    rel_pos = x_t - x_s
    invariant_1 = torch.einsum(invariant_1_einsum_eq, rel_pos, n_t)[..., None] * 1.     # rel_pos_component_along_n_t
    invariant_2 = (rel_pos[:, None, :] - invariant_1 * n_t).norm(dim=-1, keepdim=True)  # rel_pos_component_orthogonal_to_n_t
    
    # Shape of next invariants between angles in the grid will be [Nt, Ns, 1]
    if grid_dim == 2:
        invariant_3 = (n_t @ n_s.t()).unsqueeze(-1)                                     # rel_ori_component_along_n_t
    else:
        invariant_3 = (n_t[0] @ n_s[0].t()).unsqueeze(-1)                               # rel_ori_component_along_n_t

    if not separable:
        # to shape [num_pos, num_n_t, num_n_s, 3]
        invariant_1 = invariant_1[:, :, None, :].expand(-1, -1, num_n_s, -1)
        invariant_2 = invariant_2[:, :, None, :].expand(-1, -1, num_n_s, -1)
        invariant_3 = invariant_3[None, :, :, :].expand(num_pos, -1, -1, -1)
        return torch.cat([invariant_1, invariant_2, invariant_3], dim=-1)
    if separable:
        # to shapes [num_pos, num_n_t, 2], [num_n_t, num_n_s, 1]
        return (torch.cat([invariant_1, invariant_2], dim=-1), invariant_3)  # ([num_pos, num_n_t, 2], [num_n_t, num_n_s, 1])





def invariant_attributes_r3s2_euler(pos_s, pos_t, eps=1e-12):
    # Spatial positions and local orientations
    x_s, n_s, x_t, n_t = pos_s[:, :3], pos_s[:, 3:], pos_t[:, :3], pos_t[:, 3:]

    # Decompose relative position in polar coordinates (relative to n_t)
    rel_pos = x_t - x_s
    invariant_1 = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)  # rel_pos_dist
    rel_pos_component_along_n_t = torch.einsum('bi,bi->b', rel_pos, n_t)[..., None] * 1.
    rel_pos_component_orthogonal_to_n_t = torch.linalg.norm(rel_pos - rel_pos_component_along_n_t * n_t, dim=-1, keepdim=True)
    invariant_2 = torch.atan2(rel_pos_component_orthogonal_to_n_t, rel_pos_component_along_n_t + eps)  # rel_pos_angle
    
    # Relative orientation as angle between vectors (use atan method for well defined grad)
    rel_ori_component_along_n_t = torch.einsum('bi,bi->b', n_s, n_t)[..., None] * 1.
    rel_ori_component_orthogonal_to_n_t = torch.linalg.norm(n_s - n_t * rel_ori_component_along_n_t, dim=-1, keepdim=True)
    invariant_3 = torch.atan2(rel_ori_component_orthogonal_to_n_t, rel_ori_component_along_n_t)  # rel_ori_angle_between_n_s_n_t

    # Concat and return attributes
    attr = torch.cat([invariant_1, invariant_2, invariant_3], dim=-1)  # [num_pos, 3]
    return attr
    

def invariant_attributes_r3s2_euler_grid(pos_s, pos_t, grid_s, grid_t=None, separable=False, eps=1e-12):
    # Spatial positions
    x_s, x_t = pos_s[..., :3], pos_t[..., :3]
    num_pos = pos_s.shape[0]
    # Orientations
    n_s = grid_s
    n_t = n_s if grid_t is None else grid_t
    # grid sizes
    num_n_s, num_n_t = n_s.shape[0], n_t.shape[0]
   
    # Decompose relative position in polar coordinates (relative to n_t), shape of each invariant is [num_pos, num_n_t, 1]
    rel_pos = x_t - x_s
    invariant_1 = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)  # rel_pos_dist
    rel_pos_component_along_n_t = torch.einsum('bi,ni->bn', rel_pos, n_t)[..., None] * 1.
    rel_pos_component_orthogonal_to_n_t = torch.linalg.norm(rel_pos[:, None, :] - rel_pos_component_along_n_t * n_t, dim=-1, keepdim=True)
    invariant_2 = torch.atan2(rel_pos_component_orthogonal_to_n_t, rel_pos_component_along_n_t + eps)  # rel_pos_angle
    
    # Shape of invariants will be [Nt, Ns, 1]
    rel_ori_component_along_n_t = torch.einsum('ti,si->ts', n_t, n_s)[..., None] * 1.
    rel_ori_component_orthogonal_to_n_t = torch.linalg.norm(n_s[None, :, :] - n_t[:, None, :] * rel_ori_component_along_n_t, dim=-1, keepdim=True)
    invariant_3 = torch.atan2(rel_ori_component_orthogonal_to_n_t, rel_ori_component_along_n_t)  # rel_ori_angle_between_n_s_n_t

    if not separable:
        invariant_1 = invariant_1[:, None, None, :].expand(-1, num_n_t, num_n_s, -1)  # [num_pos, num_n_t, num_n_s, 1]
        invariant_2 = invariant_2[:, :, None, :].expand(-1, -1, num_n_s, -1)  # [num_pos, num_n_t, num_n_s, 1]
        invariant_3 = invariant_3[None, :, :, :].expand(num_pos, -1, -1, -1)  # [num_pos, num_n_t, num_n_s, 1]
        attr = torch.cat([invariant_1, invariant_2, invariant_3], dim=-1) # [num_pos, num_n_t, num_n_s, 1]
    if separable:
        invariant_1 = invariant_1[:, None, :].expand(-1, num_n_t, -1)  # [num_pos, num_n_t, 1]
        attr = (torch.cat([invariant_1, invariant_2], dim=-1), invariant_3)  # ([num_pos, num_n_t, 2], [num_n_t, num_n_s, 1])
    return attr