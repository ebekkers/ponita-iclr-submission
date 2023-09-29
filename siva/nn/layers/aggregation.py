from torch_geometric.nn import MessagePassing


class AggregationLayer(MessagePassing):
    def __init__(self, aggr='mean'):
        super().__init__(aggr=aggr)

    def forward(self, attr, edge_index):
        
        out = self.propagate(edge_index, attr=attr)

        return out

    def message(self, attr):
        return attr