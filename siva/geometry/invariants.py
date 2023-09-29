import torch



def invariant_attributes(pos_s, pos_t=None, grid_s=None, grid_t=None, separable=False):
    # _s: source
    # _t: target

    # Check dimension
    dim_pos = pos_s.shape[-1]
    dim_grid = grid_s.shape[-1] if grid_s is not None else 0

    if dim_pos == 3 and dim_grid == 0:
        # R3
        raise NotImplementedError("R3 invariants not implemented yet...")
    elif dim_pos == 6:
        # R3xS2
        return invariant_attributes_r3s2(pos_s, pos_t)
    elif dim_pos == 3 and dim_grid == 3 :
        # R3xS2 (grid mode)
        return invariant_attributes_r3s2_grid(pos_s, pos_t, grid_s, grid_t, separable)
    elif (dim_pos == 3 and dim_grid == 9) or dim_pos == 12:
        # R3xSO3 mode
        raise NotImplementedError("R3xSO3 invariants not implemented yet...")


def invariant_attributes_r3s2(pos_s, pos_t, eps=1e-12):
    # _s: source, _t: target

    # Spatial positions and local orientations
    x_s, n_s, x_t, n_t = pos_s[:, :3], pos_s[:, 3:], pos_t[:, :3], pos_t[:, 3:]

    # Decompose relative position in polar coordinates (relative to n_t)
    rel_pos = x_t - x_s
    rel_pos_dist = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)  # [num_pos, 1]
    rel_pos_component_along_n_t = torch.einsum('bi,bi->b', rel_pos, n_t)[..., None]
    rel_pos_component_orthogonal_to_n_t = torch.linalg.norm(rel_pos - rel_pos_component_along_n_t * n_t, dim=-1, keepdim=True)
    rel_pos_angle = torch.atan2(rel_pos_component_orthogonal_to_n_t, rel_pos_component_along_n_t + eps)  # [num_pos, 1]
    
    # Relative orientation as angle between vectors (use atan method for well defined grad)
    rel_ori_component_along_n_t = torch.einsum('bi,bi->b', n_s, n_t)[..., None]
    rel_ori_component_orthogonal_to_n_t = torch.linalg.norm(n_s - n_t * rel_ori_component_along_n_t, dim=-1, keepdim=True)
    rel_ori_angle_between_n_s_n_t = torch.atan2(rel_ori_component_orthogonal_to_n_t, rel_ori_component_along_n_t)  # [num_pos, 1]

    # Concat and return attributes
    attr = torch.cat([rel_pos_dist, rel_pos_angle, rel_ori_angle_between_n_s_n_t], dim=-1)  # [num_pos, 3]
    return attr
    
def invariant_attributes_r3s2_grid(pos_s, pos_t, grid_s, grid_t = None, separable = False, eps=1e-12):
    # _s: source, _t: target

    # Spatial positions
    x_s, x_t = pos_s[..., :3], pos_t[..., :3]
    num_pos = pos_s.shape[0]
    # Orientations
    n_s = grid_s
    n_t = n_s if grid_t is None else grid_t
    # grid sizes
    num_n_s, num_n_t = n_s.shape[0], n_t.shape[0]
   
    # Decompose relative position in polar coordinates (relative to n_t)
    rel_pos = x_t - x_s
    rel_pos_dist = torch.linalg.norm(rel_pos, dim=-1, keepdim=True) # [num_pos, 1]
    rel_pos_component_along_n_t = torch.einsum('bi,ni->bn', rel_pos, n_t)[..., None]
    rel_pos_component_orthogonal_to_n_t = torch.linalg.norm(rel_pos[:, None, :] - rel_pos_component_along_n_t * n_t, dim=-1, keepdim=True)  # [num_pos, num_n_t, 1]
    rel_pos_angle = torch.atan2(rel_pos_component_orthogonal_to_n_t, rel_pos_component_along_n_t + eps)  # [num_pos, num_n_t, 1]
    
    # Shape of invariants will be [Ns, Nt, 1]
    rel_ori_component_along_n_t = torch.einsum('ti,si->ts', n_t, n_s)[..., None]  # [num_n_t, num_n_s, 1]
    rel_ori_component_orthogonal_to_n_t = torch.linalg.norm(n_s[None, :, :] - n_t[:, None, :] * rel_ori_component_along_n_t, dim=-1, keepdim=True)
    rel_ori_angle_between_n_s_n_t = torch.atan2(rel_ori_component_orthogonal_to_n_t, rel_ori_component_along_n_t)  # [num_n_t, num_n_s, 1]

    if not separable:
        rel_pos_dist = rel_pos_dist[:, None, None, :].expand(-1, num_n_t, num_n_s, -1)  # [num_pos, num_n_t, num_n_s, 1]
        rel_pos_angle = rel_pos_angle[:, :, None, :].expand(-1, -1, num_n_s, -1)  # [num_pos, num_n_t, num_n_s, 1]
        rel_ori_angle_between_n_s_n_t = rel_ori_angle_between_n_s_n_t[None, :, :, :].expand(num_pos, -1, -1, -1)  # [num_pos, num_n_t, num_n_s, 1]
        attr = torch.cat([rel_pos_dist, rel_pos_angle, rel_ori_angle_between_n_s_n_t], dim=-1) # [num_pos, num_n_t, num_n_s, 1]
    if separable:
        rel_pos_dist = rel_pos_dist[:, None, :].expand(-1, num_n_t, -1)  # [num_pos, num_n_t, 1]
        attr = (torch.cat([rel_pos_dist, rel_pos_angle], dim=-1), rel_ori_angle_between_n_s_n_t)  # ([num_pos, num_n_t, 2], [num_n_t, num_n_s, 1])
    return attr


# b = 1000
# pos_s = torch.randn(b, 6)
# pos_t = torch.randn(b, 6)
# pos_s[:,3:]/=torch.linalg.norm(pos_s[:,3:],dim=-1,keepdim=True)
# pos_t[:,3:]/=torch.linalg.norm(pos_t[:,3:],dim=-1,keepdim=True)
# # pos_t[:,3:]=pos_s[:,3:]
# # pos_t[:,:3] = pos_s[:,:3] + pos_s[:,3:]

# attr = invariant_attributes(pos_s, pos_t)

# print('direct', attr[:,0].min(), attr[:,1].min(), attr[:,2].min())
# print('direct', attr[:,0].max(), attr[:,1].max(), attr[:,2].max())


# # grid mode
# pos_s = torch.randn(b, 3)
# pos_t = torch.randn(b, 3)
# num_n_s, num_n_t = 20, 30
# grid_s = torch.randn(num_n_s, 3)
# grid_t = torch.randn(num_n_t, 3)
# grid_s /= torch.linalg.norm(grid_s, dim=-1, keepdim=True)
# grid_t /= torch.linalg.norm(grid_t, dim=-1, keepdim=True)

# attr = invariant_attributes(pos_s, pos_t, grid_s, grid_t, separable=False)

# print('grid', attr[:,:,:, 0].min(), attr[:,:,:, 1].min(), attr[:,:,:,2].min())
# print('grid', attr[:,:,:, 0].max(), attr[:,:,:, 1].max(), attr[:,:,:,2].max())


# attr_x, attr_n = invariant_attributes(pos_s, pos_t, grid_s, grid_t, separable=True)


# print('grid sep', attr_x[:,:,0].min(), attr_x[:,:,1].min(), attr_n[:,:,0].min())
# print('grid sep', attr_x[:,:,0].max(), attr_x[:,:,1].max(), attr_n[:,:,0].max())

