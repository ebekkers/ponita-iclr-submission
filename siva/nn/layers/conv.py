import torch
import torch_geometric
from torch.nn import LayerNorm


class Conv(torch_geometric.nn.MessagePassing):
    """
    Implementation of a convolutional layer for point clouds. The forward assumes pair-wise attributes that condition the convolution kernel to be precomputed and stored in eddge_attr.
    The kernel is then defined as k(a_ij) = linear(a_ij) which returns a (num_edges, out_dim, in_dim) dimensional tensor or only a vector-valued batch (num_edges, out_dim) if 
    depthwise=`True`, in which case no "channel-mixing" takes place in the convolution.

    Args:
        in_dim (int): The input dimension of the convolutional layer.
        out_dim (int): The output dimension of the convolutional layer.
        attr_dim (int): The dimensionality of the edge features.
        bias (bool, optional): If set to `True`, a bias term will be added to the layer. (default: `True`)
        aggr (str, optional): The aggregation method to use for combining messages. (default: `"add"`)
        depthwise (bool, optional): If set to `True`, a depthwise convolution will be performed. In this case, the input and output dimensions must be the same. (default: `False`)
    """
    def __init__(self, in_dim, out_dim, attr_dim, bias=True, aggr="add", depthwise=False):
        super().__init__(node_dim=0, aggr=aggr)

        self.in_dim, self.out_dim = in_dim, out_dim
        self.depthwise = depthwise

        if self.depthwise:
            self.einsum_eq = '...o,...o->...o'
            if in_dim != out_dim:
                raise ValueError(f"When depthwise=True in- and output dimensions should be the same")
            self.kernel = torch.nn.Linear(attr_dim, out_dim, bias=False)
        else:
            self.einsum_eq = '...oi,...i->...o'
            self.kernel = torch.nn.Linear(attr_dim, out_dim * in_dim, bias=False)

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_dim))
            self.bias.data.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_attr, edge_index, batch_target=None, batch_source=None):
        """
        Performs a forward pass of the convolutional layer.

        Args:
            x (Tensor): The node feature matrix of shape `(num_nodes, in_dim)`.
            edge_attr (Tensor): The edge feature matrix of shape `(num_edges, attr_dim)`.
            edge_index (LongTensor): The edge index tensor of shape `(2, num_edges)`.
            batch_target (LongTensor, optional): A tensor that assigns each edge to a target batch index (for bipartite graphs). This tensor has shape `(num_edges,)`. (default: `None`)
            batch_source (LongTensor, optional): A tensor that assigns each edge to a source batch index (for bipartite graphs). This tensor has shape `(num_edges,)`. (default: `None`)

        Returns:
            Tensor: The output feature matrix of shape `(num_nodes, out_dim)`.
        """
        if batch_source is not None and batch_target is not None:
            size = (batch_source.shape[0], batch_target.shape[0])
        else:
            size = None
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size = size)

    def message(self, x_j, edge_attr):
        kernel = self.kernel(edge_attr)
        if not self.depthwise:
            kernel = kernel.unflatten(-1, (self.out_dim, self.in_dim))
        return torch.einsum(self.einsum_eq, kernel, x_j) * 1.
    
    def update(self, message_aggr):
        out = message_aggr
        if self.bias is not None:
            out += self.bias
        return out
