import torch
import torch_geometric
from torch_geometric.utils import softmax
import math


class FiberConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attr_dim, bias=True, depthwise=False):
        super().__init__()

        self.in_dim, self.out_dim = in_dim, out_dim
        self.depthwise = depthwise

        if depthwise:
            self.einsum_eq = 'tso,bso->bto'
            if in_dim != out_dim:
                raise ValueError(f"When depthwise=True in- and output dimensions should be the same")
            self.kernel_fiber = torch.nn.Linear(attr_dim, in_dim, bias=False)
        else:
            self.einsum_eq = 'tsoi,bsi->bto'
            self.kernel_fiber = torch.nn.Linear(attr_dim, out_dim * in_dim, bias=False)

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_dim))
            self.bias.data.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_attr_fiber):
        kernel_fiber = self.kernel_fiber(edge_attr_fiber)
        if not self.depthwise:
            kernel_fiber = kernel_fiber.unflatten(-1, (self.out_dim, self.in_dim))
        out = torch.einsum(self.einsum_eq, kernel_fiber, x) * 1.
        if self.bias is not None:
            return out + self.bias
        else:  
            return out


class TransformerConv(torch_geometric.nn.MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        attr_dim,
        heads = 1,
        concat = True,
        beta = False,
        dropout = 0.,
        edge_dim = None,
        bias = True,
        root_weight = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        
        self.lin_key = FiberConv(in_channels[0], heads * out_channels, attr_dim, depthwise=False)
        self.lin_query = FiberConv(in_channels[0], heads * out_channels, attr_dim, depthwise=False)
        self.lin_value = FiberConv(in_channels[0], heads * out_channels, attr_dim, depthwise=False)  
        self.lin_edge = self.register_parameter('lin_edge', None)

    
        self.lin_skip = torch.nn.Linear(in_channels[1], out_channels, bias=bias)
        if self.beta:
            self.lin_beta = torch.nn.Linear(3 * out_channels, 1, bias=False)
        else:
            self.lin_beta = self.register_parameter('lin_beta', None)

    def forward(self, x, edge_index,edge_attr_fiber):

        H, C = self.heads, self.out_channels
        N = x.shape[1]

        x = (x, x)

        query = self.lin_query(x[1], edge_attr_fiber).view(-1, H, N * C)
        key = self.lin_key(x[0], edge_attr_fiber).view(-1, H, N * C)
        value = self.lin_value(x[0], edge_attr_fiber).view(-1, N, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=None, size=None)

        if self.concat:
            out = out.view(-1, N, self.heads * self.out_channels)
        else:
            out = out.mean(dim=-2)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        return out


    def message(self, query_i, key_j, value_j, index):

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)  # [-1, H]
        alpha = softmax(alpha, index)
        self._alpha = alpha

        out = value_j  # [-1, N, H, C]

        out = out * alpha.view(-1, 1, self.heads, 1)
        return out
