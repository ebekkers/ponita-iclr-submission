import torch


# def rel_pos_r3s2_backup(pos, edge_index):
#     pos1, pos2 = pos[edge_index[1]], pos[edge_index[0]]
    
#     # Unpack left position and corresponding rotation matrix
#     x1 = pos1[..., :3]
#     n1 = pos1[..., 3:]
#     R1_T = n_to_matrix(n1).transpose(-1,-2)
#     # Unpack right position
#     x2 = pos2[..., :3]
#     n2 = pos2[..., 3:]

#     # Inverse (see transpose) group action of g1 = (x1,R1) acting on (x2, n2)
#     rel_x = torch.matmul(R1_T, (x2 - x1)[...,None])[...,0]
#     rel_n = torch.matmul(R1_T, n2[...,None])[...,0]

#     # Prepare and return output
#     rel_pos = torch.cat([rel_x, rel_n], dim=-1)

#     return rel_pos

def rel_pos_r3s2(pos_s, pos_t):
    
    # Unpack left position and corresponding rotation matrix
    x_t = pos_t[..., :3]
    n_t = pos_t[..., 3:]
    R_t_T = n_to_matrix(n_t).transpose(-1,-2)

    # Unpack right position
    x_s = pos_s[..., :3]
    source_in_R3 = pos_s.shape[-1] == 3
    n_s = n_t if source_in_R3 else pos_s[...,3:]  # if source_in_R3 just use n_t as a place holder

    # Inverse (see transpose) group action of target g_t = (x_t,R_t) acting on source position (x_s, n_s)
    rel_x = torch.matmul(R_t_T, (x_s - x_t)[...,None])[...,0]
    rel_n = torch.matmul(R_t_T, n_s[...,None])[...,0]

    # Prepare and return output
    rel_pos = torch.cat([rel_x, rel_n], dim=-1)

    return rel_pos[:,:3] if source_in_R3 else rel_pos


def log_r3s2(rel_pos, return_invariants=False, eps=1e-6):

    # Unpack
    x = rel_pos[:,:3]
    source_in_R3 = rel_pos.shape[-1] == 3
    
    if source_in_R3:
        c123 = x
        angle = 0
        axis = torch.zeros_like(c123)
    else:
        # compute log
        n = rel_pos[...,3:]

        # Prepare log of rotation part
        axis, angle = n_to_axis_angle(n)

        # Prepare log of spatial part
        w0 = axis[..., 0, None]
        w1 = axis[..., 1, None]

        angle_cot = 0.5 * angle / torch.tan(0.5 * angle)
        angle_cot[angle < eps] = 1.
        angle_cot[angle > torch.pi - eps] = 0.

        matrix = torch.cat([
            1 - (1 - angle_cot) * (w1 ** 2),
            (1 - angle_cot) * w0 * w1,
            - angle * w1 / 2.,
            (1 - angle_cot) * w0 * w1,
            1 - (1 - angle_cot) * (w0 ** 2),
            angle * w0 / 2.,
            angle * w1 / 2.,
            - angle * w0 / 2.,
            1 + (1 - angle_cot) * ( - (angle ** 2) * (w0 ** 2) - (angle ** 2) * (w1 ** 2)) / (eps + angle ** 2)
        ], dim=-1).unflatten(-1, (3,3))
        c123 = torch.matmul(matrix, x[...,None])[...,0]

    if return_invariants:
        # Return the invariants of the log vector
        c12 = torch.linalg.norm(c123[...,:2], dim=-1, keepdim=True)
        c3 = c123[...,2:3]
        c45 = angle
        return torch.cat([c12,c3],dim=-1) if source_in_R3 else torch.cat([c12,c3,c45],dim=-1)
    else:
        # Return log vector
        c456 = axis * angle
        return torch.cat([c123,c456], -1)


# def edge_attr_r3s2_bu(pos, edge_index, node_attr=None, eps=1e-6, lie_algebra=True):  # Invariants in manifold representation (not Lie algebra)
#     rel_pos = rel_pos_r3s2(pos, edge_index)
#     if lie_algebra:
#         edge_attr = [log_r3s2(rel_pos, return_invariants=True)]
#     else:
#         edge_attr = [torch.linalg.norm(rel_pos[:,:2],dim=-1, keepdim=True), rel_pos[:,2:3], rel_pos[:,5:]]

#     if node_attr is not None:
#         node_attr1, node_attr2 = node_attr[edge_index[1]], node_attr[edge_index[0]]
#         edge_attr = [node_attr1, node_attr2] + edge_attr
#     edge_attr = torch.cat(edge_attr, dim=-1)

#     return edge_attr


def edge_attr_r3s2(pos_source, pos_target, lie_algebra=True, eps=1e-6):  # Invariants in manifold representation (not Lie algebra)
    rel_pos = rel_pos_r3s2(pos_source, pos_target)
    if lie_algebra:
        edge_attr = [log_r3s2(rel_pos, return_invariants=True, eps=eps)]
    else:
        edge_attr = [torch.linalg.norm(rel_pos[:,:2],dim=-1, keepdim=True), rel_pos[:,2:3], rel_pos[:,5:]]
    edge_attr = torch.cat(edge_attr, dim=-1)

    return edge_attr


def r3_edge_to_r3s2(pos, x, batch, edge_index):
    pos1 = pos[edge_index[1]]
    pos2 = pos[edge_index[0]]
    x1 = x[edge_index[1]]
    x2 = x[edge_index[0]]
    new_batch = batch[edge_index[1]]

    rel_pos = pos2 - pos1
    dist = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
    direction = rel_pos / dist

    new_pos = torch.cat([pos1, direction],-1)
    new_x = torch.cat([x1, x2], dim=-1)
    # siva = dist

    return new_pos, new_x, new_batch




def n_to_axis_angle(n):
    eps = 1e-6
    angle = torch.arccos( n[...,2].clamp(-1+eps, 1-eps) )
    axis = torch.stack([-n[..., 1] + eps, n[...,0], torch.zeros_like(n[...,0])], dim=-1)
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
    return axis, angle[..., None]


def axis_angle_to_matrix(axis, angle, eps=1e-6):
        # Adapted from https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/core/conversions.html#angle_axis_to_rotation_matrix
        k_one = 1.0
        theta = angle
        wxyz = axis
        wx, wy, wz = torch.chunk(wxyz, 3, dim=-1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1)
        return rotation_matrix.unflatten(-1, (3, 3))


def n_to_matrix(n):
    axis, angle = n_to_axis_angle(n)
    return axis_angle_to_matrix(axis, angle)


def random_rotation():
    axis = torch.randn(1, 1, 3)
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
    angle = torch.rand(1, 1, 1) * torch.pi
    matrix = axis_angle_to_matrix(axis, angle)
    return matrix[0,0]



if __name__ == "__main__":

    B, N, d = 32, 100, 3

    x = torch.randn(B, N, d)
    n = torch.randn(B, N, d)
    n = n / torch.linalg.norm(n, dim=-1, keepdim=True)
    pos = torch.cat([x, n], dim=-1)

    axis, angle = n_to_axis_angle(n)
    R = axis_angle_to_matrix(axis, angle)

    n2 = R @ torch.tensor([0,0,1]).type_as(R)
    
    print()
    print('Is the direction so rotation conversion is 1e-5 accurate?')
    print('   ', torch.allclose(n, n2, atol=1e-5))

    rel_pos = rel_pos_r3s2(pos[:,:,None,:], pos[:,None,:,:])
    log_rel_pos = log_r3s2(rel_pos)
    inv = rel_pos_invariants_r3s2(log_rel_pos)


    random_R = random_rotation()
    x2 = torch.einsum('ij,...i->...j', random_R, x)
    n2 = torch.einsum('ij,...i->...j', random_R, n)
    pos2 = torch.cat([x2, n2], dim=-1)

    rel_pos2 = rel_pos_r3s2(pos2[:,:,None,:], pos2[:,None,:,:])
    log_rel_pos2 = log_r3s2(rel_pos2)
    inv2 = rel_pos_invariants_r3s2(log_rel_pos2)

    print()
    print('Are the invariants "truely" invariant, i.e. the same up to 1e-3?')
    print('   ', torch.allclose(inv, inv2, atol=1e-3))

