import torch
from siva.nn.layers.aggregation import AggregationLayer


# aggregator = AggregateAttributes('mean')
aggregator = AggregationLayer('sum')

# def rel_pos_se3(pos, edge_index):
#     pos1, pos2 = pos[edge_index[1]], pos[edge_index[0]]
    
#     # Unpack left position and corresponding rotation matrix
#     x1 = pos1[..., :3]
#     R1 = pos1[..., 3:].unflatten(-1,(3,3))
#     R1_T = R1.transpose(-1,-2)
#     # Unpack right position
#     x2 = pos2[..., :3]
#     R2 = pos2[..., 3:].unflatten(-1, (3,3))

#     # Inverse (see transpose) group action of g1 = (x1,R1) acting on (x2, n2)
#     rel_x = torch.matmul(R1_T, (x2 - x1)[...,None])[...,0]
#     rel_R = torch.einsum('bij,bjk->bik', R1_T, R2)

#     # Prepare and return output
#     rel_pos = torch.cat([rel_x, rel_R.flatten(-2,-1)], dim=-1)

#     return rel_pos


def rel_pos_se3(pos_s, pos_t):
    # pos1, pos2 = pos[edge_index[1]], pos[edge_index[0]]
    
    # Unpack left position and corresponding rotation matrix
    x_t = pos_t[..., :3]
    R_t = pos_t[..., 3:].unflatten(-1,(3,3))
    R_t_T = R_t.transpose(-1,-2)
    # Unpack right position
    x_s = pos_s[..., :3]
    source_in_R3 = pos_s.shape[-1] == 3
    R_s = R_t if source_in_R3 else pos_s[...,3:].unflatten(-1, (3,3))  # if source_in_R3 just use R_t as a place holder

    # Inverse (see transpose) group action of target g_t = (x_t,R_t) acting on source (x_s, n_s)
    rel_x = torch.matmul(R_t_T, (x_s - x_t)[...,None])[...,0]
    rel_R = torch.einsum('bij,bjk->bik', R_t_T, R_s)

    # Prepare and return output
    rel_pos = torch.cat([rel_x, rel_R.flatten(-2,-1)], dim=-1)

    return rel_pos[:,:3] if source_in_R3 else rel_pos


def log_se3(rel_pos, return_invariants=False, eps=1e-6):

    # Unpack
    x = rel_pos[...,:3]
    source_in_R3 = rel_pos.shape[-1] == 3

        
    if source_in_R3:
        c123 = x
        angle = 0
        axis = torch.zeros_like(c123)
    else:
        R = rel_pos[...,3:].unflatten(-1,(3,3))

        # Prepare log of rotation part
        axis, angle = rotation_matrix_to_axis_angle(R)

        # Check if matrix exp again gives the original rotation matrix
        # A1 = torch.tensor([[0, 0, 0],[0,0,-1], [0,1,0]]).type_as(R)
        # A2 = torch.tensor([[0, 0, 1],[0,0,0], [-1,0,0]]).type_as(R)
        # A3 = torch.tensor([[0, -1, 0],[1,0,0], [0,0,0]]).type_as(R)
        # A = A1[None,:,:] * axis[:, 0][:,None,None] * angle[:,None] + A2[None,:,:] * axis[:, 1][:,None,None] * angle[:,None] + A3[None,:,:] * axis[:, 2][:,None,None] * angle[:,None]
        # torch.allclose(R, torch.linalg.matrix_exp(A), rtol=1, atol=1e-4)

        # Prepare log of spatial part
        w0 = axis[..., 0, None]
        w1 = axis[..., 1, None]
        w2 = axis[..., 2, None]

        eps=1-14
        angle_cot = 0.5 * angle / torch.tan(0.5 * angle)
        angle_cot[angle < eps] = 1.
        angle_cot[angle > torch.pi - eps] = 0.

        matrix = torch.cat([
            1 + (-1 + angle_cot) * (w1 ** 2) + (-1 + angle_cot) * (w2 ** 2),
            w0 * (w1 - angle_cot * w1) + angle * w2 / 2.,
            - angle * w1 / 2. - (-1 + angle_cot) * w0 * w2,
            w0 * (w1 - angle_cot * w1) - angle * w2 / 2.,
            1 + (-1 + angle_cot) * (w0 ** 2) + (-1 + angle_cot) * (w2 ** 2),
            angle * w0 / 2. - (-1 + angle_cot) * w1 * w2,
            angle * w1 / 2. - (-1 + angle_cot) * w0 * w2,
            -angle * w0 / 2. - (-1 + angle_cot) * w1 * w2,
            1 + (-1 + angle_cot) * (w0 ** 2) + (-1 + angle_cot) * (w1 ** 2)
        ], dim=-1).unflatten(-1, (3,3))
        c123 = torch.matmul(matrix, x[...,None])[...,0]

    # Return log vector
    c456 = axis * angle

    # A1 = torch.tensor([[0,0,0,1],[0,0,0,0],[0,0,0,0], [0,0,0,0]]).type_as(R)[None,:,:]
    # A2 = torch.tensor([[0,0,0,0],[0,0,0,1],[0,0,0,0], [0,0,0,0]]).type_as(R)[None,:,:]
    # A3 = torch.tensor([[0,0,0,0],[0,0,0,0],[0,0,0,1], [0,0,0,0]]).type_as(R)[None,:,:]
    # A4 = torch.tensor([[0, 0, 0,0],[0,0,-1,0], [0,1,0,0], [0,0,0,0]]).type_as(R)[None,:,:]
    # A5 = torch.tensor([[0, 0, 1,0],[0,0,0,0], [-1,0,0,0], [0,0,0,0]]).type_as(R)[None,:,:]
    # A6 = torch.tensor([[0, -1, 0,0],[1,0,0,0], [0,0,0,0], [0,0,0,0]]).type_as(R)[None,:,:]
    # A = A1 * c123[:,0,None,None] + A2 * c123[:,1,None,None] + A3 * c123[:,2,None,None] \
    #      + A4 * c456[:,0,None,None] + A5 * c456[:,1,None,None] + A6 * c456[:,2,None,None]
    # xR = torch.linalg.matrix_exp(A)
    # torch.allclose(R, xR[:,:3,:3], rtol=1, atol=1e-4)
    # torch.allclose(x, xR[:,:3,3], rtol=1, atol=1e-1)

    return torch.cat([c123],dim=-1) if source_in_R3 else torch.cat([c123,c456],dim=-1)


# def edge_attr_se3(pos, edge_index, node_attr=None, eps=1e-6, lie_algebra=True):  # Invariants in manifold representation (not Lie algebra)
#     rel_pos = rel_pos_se3(pos, edge_index)
#     if lie_algebra:
#         edge_attr = [log_se3(rel_pos, return_invariants=True)]
#     else:
#         edge_attr = [rel_pos]
#         # edge_attr = [torch.stack([rel_pos[:,0],rel_pos[:,1],rel_pos[:,2],rel_pos[:,3],rel_pos[:,4],rel_pos[:,5],rel_pos[:,7],rel_pos[:,8],rel_pos[:,11]],dim=-1)]

#     if node_attr is not None:
#         node_attr1, node_attr2 = node_attr[edge_index[1]], node_attr[edge_index[0]]
#         edge_attr = [node_attr1, node_attr2] + edge_attr
#     edge_attr = torch.cat(edge_attr, dim=-1)

#     return edge_attr

def edge_attr_se3(pos_source, pos_target, lie_algebra=True, eps=1e-6):  # Invariants in manifold representation (not Lie algebra)
    rel_pos = rel_pos_se3(pos_source, pos_target)
    if lie_algebra:
        edge_attr = [log_se3(rel_pos, return_invariants=True, eps=eps)]
    else:
        edge_attr = [rel_pos]
        # edge_attr = [torch.stack([rel_pos[:,0],rel_pos[:,1],rel_pos[:,2],rel_pos[:,3],rel_pos[:,4],rel_pos[:,5],rel_pos[:,7],rel_pos[:,8],rel_pos[:,11]],dim=-1)]
    edge_attr = torch.cat(edge_attr, dim=-1)

    return edge_attr

def r3_edge_to_se3(pos, x, batch, edge_index, radius = 5.):
    
    pos1 = pos[edge_index[1]] # Center point 
    pos2 = pos[edge_index[0]] # Neighbor point
    pos2_weight = dist_window( torch.linalg.norm(pos2 - pos1, dim=-1, keepdim=True), radius)
    pos2_weight = pos2_weight / aggregator(pos2_weight, edge_index)[edge_index[1]]
    pos0 = aggregator(pos2 * pos2_weight, edge_index)[edge_index[1]]  # Center of mass associcated with all neighbors of center point
    x1 = x[edge_index[1]]
    # x2 = x[edge_index[0]]  # todo: potentially expand to an x0 (mean of neigh features)
    new_batch = batch[edge_index[1]]

    # Main direction
    rel_pos_12 = pos2 - pos1
    dist_12 = torch.linalg.norm(rel_pos_12, dim=-1, keepdim=True)
    direction_a = rel_pos_12 / dist_12

    # Direction towards center of mass
    rel_pos_10 = pos0 - pos1  # [B, 3]
    rel_pos_10 = rel_pos_10 - torch.einsum('bi,bi->b', rel_pos_10, direction_a)[..., None] * direction_a
    dist_10 = torch.linalg.norm(rel_pos_10, dim=-1, keepdim=True)
    direction_b = rel_pos_10 / dist_10

    # Orthogonal direction (via cross product)
    direction_c = torch.linalg.cross(direction_a, direction_b, dim=-1)

    # Now we have a frame (rotation matrix) at each point
    R = torch.stack([direction_a, direction_b, direction_c], dim=-1)


    new_pos = torch.cat([pos1, R.flatten(-2,-1)],-1)  # 3 + 9 dimensional vector
    # new_x = torch.cat([x1, x2], dim=-1)
    new_x = x1
    # siva = dist_12

    return new_pos, new_x, new_batch

def dist_window(dist, max_dist, min_dist=0.):
    range = max_dist - min_dist
    return 0.5 * (torch.cos(torch.pi * (dist - min_dist) / range) + 1)




# Adapted from https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/core/conversions.html#rotation_matrix_to_angle_axis
def rotation_matrix_to_axis_angle(rotation_matrix):
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_axis_angle(quaternion)


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    # mask_c1 = mask_d2 * (1 - mask_d0_d1)
    mask_c1 = mask_d2 * ~(mask_d0_d1)
    # mask_c2 = (1 - mask_d2) * mask_d0_nd1
    mask_c2 = ~(mask_d2) * mask_d0_nd1
    # mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)
    mask_c3 = ~(mask_d2) * ~(mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def quaternion_to_axis_angle(quaternion: torch.Tensor, eps=1e-6) -> torch.Tensor:
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta.clamp(eps))  # sin_squared_theta should have values between 0 and 1
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / (sin_theta.clamp(eps))  # sin_theta should have  values between 0 and 1
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    angle: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    axis[..., 0] += q1
    axis[..., 1] += q2
    axis[..., 2] += q3
    return axis, angle[..., None]