import torch
import torch.nn.functional as F
import torch_geometric as tg
import scipy
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm


qm9_targets = [
    "mu",
    "alpha",
    "homo",
    "lumo",
    "delta",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
    "U0_atom",
    "U_atom",
    "H_atom",
    "G_atom",
    "A",
    "B",
    "C",
]

class TargetGetter(BaseTransform):
    def __init__(self, target):
        self.target = target
        try:
            self.target_idx = qm9_targets.index(target)
        except:
            ValueError(
                "Target could not be found:", target, "Try one of these:", qm9_targets
            )

    def __call__(self, graph):
        graph.y = graph.y[:, self.target_idx]
        return graph


class OneHotTransform(BaseTransform):
    def __init__(self, k=None):
        super().__init__()
        self.k = k

    def __call__(self, graph):
        if self.k is None:
            graph.x = F.one_hot(graph.z).float()
        else:
            graph.x = F.one_hot(graph.z, self.k).squeeze().float()

        return graph


class Kcal2meV(BaseTransform):
    def __init__(self):
        # Kcal/mol to meV
        self.conversion = 43.3634

    def __call__(self, graph):
        graph.energy = graph.energy * self.conversion
        graph.force = graph.force * self.conversion
        return graph


class EDM(BaseTransform):
    def __call__(self, graph):
        pos = graph.pos

        edm = pos[:, None, :] - pos[None, :, :]
        edm = torch.linalg.vector_norm(edm, dim=-1)
        graph.edm = edm
        return graph


def add_edm(graph):
    lens, sections = get_sections(graph)
    max_len = lens.max()

    pos = graph.pos.tensor_split(sections)
    EDM = []
    for p in pos:
        EDM.append(torch.linalg.vector_norm(p[:, None, :] - p[None, :, :], dim=-1))

    for i in range(len(EDM)):
        edm = EDM[i]
        l = edm.size(0)
        diff = max_len - l
        if diff > 0:
            edm = torch.cat((edm, torch.zeros(diff, l)), dim=0)
            EDM[i] = torch.cat((edm, torch.zeros(max_len, diff)), dim=1)

    EDM = torch.stack(EDM, dim=0)
    EDM = EDM.unsqueeze(-1)
    graph.EDM = EDM

    return graph


def pad_graph(graph, keys):
    """Pad tensors in a graph object"""
    assert len(keys) == len(set(keys)), "You have duplicate keys"

    # Find how to split the graph dataset into lists
    _, sections = get_sections(graph)

    for key in keys:
        try:
            data = graph[key].tensor_split(sections)
            graph[key] = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        except KeyError:
            print("Key not found:", key)
    return graph


def get_sections(graph):
    _, lens = graph.batch.unique(return_counts=True)
    graph.mask = create_mask(lens)
    sections = lens[:-1].long().cumsum(0).cpu()
    return lens, sections


def create_mask(lens):
    """Create batched masks"""
    max_len = max(lens)
    mask = torch.arange(max_len, device=lens.device)[None, :] >= lens[:, None]
    return mask


if __name__ == "__main__":
    import torch_geometric as tg
    from torch_geometric.loader import DataLoader

    dataset = tg.datasets.QM9("data")[:10]
    loader = DataLoader(dataset, batch_size=3, shuffle=True)

    for graph in loader:
        # print(graph.x)

        graph = add_edm(graph)
        # batch = pad_graph(graph, ["x"])
        # print(graph)
        # print(graph.mask)
        # print(graph.x)

        break
