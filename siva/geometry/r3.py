import torch


def rel_pos_r3(pos, edge_index):
    pos1, pos2 = pos[edge_index[1]], pos[edge_index[0]]
    rel_pos = pos2 - pos1
    return rel_pos


def log_r3(rel_pos, return_invariants=False):
    if return_invariants:
        return torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
    else:
        return rel_pos


def edge_attr_r3(pos, edge_index, node_attr=None, eps=1e-6, lie_algebra=False):  # Invariants in manifold representation (not Lie algebra)
    rel_pos = rel_pos_r3(pos, edge_index)
    edge_attr = [log_r3(rel_pos, return_invariants=True)]
    if node_attr is not None:
        node_attr1, node_attr2 = node_attr[edge_index[1]], node_attr[edge_index[0]]
        edge_attr = [node_attr1, node_attr2] + edge_attr
    edge_attr = torch.cat(edge_attr, dim=-1)
    return edge_attr
