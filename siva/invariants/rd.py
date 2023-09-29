import torch


def invariant_attributes_rd(pos_s, pos_t):
    edge_dist = torch.linalg.norm(pos_s - pos_t, dim=-1, keepdim=True)  # [num_nodes, 1]
    return edge_dist
