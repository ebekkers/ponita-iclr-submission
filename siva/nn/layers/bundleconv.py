import torch
import torch_geometric


class BundleConv(torch_geometric.nn.MessagePassing):
    """
    The `BundleConv` module implements a group convolution on a point cloud of "fibers". That is, we consider a point cloud
    on R^n, together with a grid of directions or rotation matrices. At each node, a fiber (vector) of features associated 
    with these grid points is stored. During convolution, the "relative meaning" of a grind point on the source site relative 
    to a grid point at the target site is considered. This "relative meaning" essentially should capture the invariant part of
    the relative SE(d) group action that maps on localized grid point to the other and is stored in the `edge_attr`. The 
    `edge_attr` linearly conditions the convolution kernel.
    
    The input `x` is then a tensor of shape (num_nodes, num_grid_points, in_dim). The edge attribute `edge_attr` now has to 
    encode for all possible interactions between the two grid points at the source and target node and is of thus of shape 
    (num_edges, num_grid_points, num_grid_points, attr_dim).

    Args:
        in_dim (int): Input feature dimensionality.
        out_dim (int): Output feature dimensionality.
        attr_dim (int): Edge feature dimensionality.
        bias (bool, optional): If set to `False`, the layer will not learn an additive bias. (default: `True`)
        aggr (string, optional): The aggregation operator to use ("add", "mean", or "max"). (default: `"add"`)
        depthwise (bool, optional): If set to `True`, applies a depthwise convolution to the input channels.
            (default: `False`)
    """
    def __init__(self, in_dim, out_dim, attr_dim, bias=True, aggr="add", depthwise=False):
        super().__init__(node_dim=0, aggr=aggr)

        self.in_dim, self.out_dim = in_dim, out_dim
        self.depthwise = depthwise

        # b positional index, t location in output/target fiber, s location in source fiber, o output channel, i input channel
        if depthwise:
            self.einsum_eq = 'btso,bso->bto'
            if in_dim != out_dim:
                raise ValueError(f"When depthwise=True in- and output dimensions should be the same")
            self.kernel = torch.nn.Linear(attr_dim, out_dim, bias=False)
        else:
            self.einsum_eq = 'btsoi,bsi->bto'
            self.kernel = torch.nn.Linear(attr_dim, out_dim * in_dim, bias=False)

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_dim))
            self.bias.data.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_attr, edge_index, batch_target=None, batch_source=None):
        """
        Performs a forward pass of the bundle convolutional layer.

        Args:
            x (torch.Tensor): Input node features of shape (num_nodes, num_grid_points, in_dim).
            edge_attr (torch.Tensor): Edge features of shape (num_edges, num_grid_points, num_grid_points, attr_dim).
            edge_index (torch.Tensor): Graph connectivity in COO format with shape (2, num_edges).
            batch_target (torch.Tensor, optional): Batch vector of target nodes of shape (num_nodes,). (default: `None`)
            batch_source (torch.Tensor, optional): Batch vector of source nodes of shape (num_nodes,). (default: `None`)

        Returns:
            out (torch.Tensor): Output node features of shape (num_nodes, num_grid_points, out_dim).
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
        out = torch.einsum(self.einsum_eq, kernel, x_j) * 1.
        return out
    
    def update(self, message_aggr):
        out = message_aggr
        if self.bias is not None:
            out += self.bias
        return out
    

class SeparableBundleConv(torch_geometric.nn.MessagePassing):
    """
    Efficient implementation of the BundleConv in which the convolution operator is separated over spatial interactions
    followed by interactions within the fibers. This requires two provide two types of edge attributes, one for the 
    spatial interactions of shape (num_edges, attr_dim) and one for the between grid point interactions of shape
    (num_grid_points, num_grid_points, attr_dim). The module is otherwise the same as BundleConv.

    The convolution kernel is then parametrized as
        k_oi(edge_attr_x,edge_attr_R) = k_o(edge_attr_x) k_oi(edge_attr_R)
    in which channel mixing thus takes place during the within fiber convolution, keeping the spatial convolution 
    light-weight.

    Args:
        in_dim (int): Input feature dimensionality.
        out_dim (int): Output feature dimensionality.
        attr_dim (int): Edge feature dimensionality.
        bias (bool, optional): If set to `False`, the layer will not learn an additive bias. (default: `True`)
        aggr (string, optional): The aggregation operator to use ("add", "mean", or "max"). (default: `"add"`)
        depthwise (bool, optional): If set to `True`, applies a depthwise convolution to the input channels.
            (default: `False`)
    """
    def __init__(self, in_dim, out_dim, attr_dim, bias=True, aggr="add", depthwise=False):
        super().__init__(node_dim=0, aggr=aggr)

        self.in_dim, self.out_dim = in_dim, out_dim
        self.depthwise = depthwise

        self.kernel_spatial = torch.nn.Linear(attr_dim, in_dim, bias=False)
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

        # self.callibrated = False
        self.register_buffer("callibrated", torch.tensor(False))

    def forward(self, x, edge_attr_spatial, edge_attr_fiber, edge_index, batch_target=None, batch_source=None):
        """
        Performs a forward pass of the separable bundle convolutional layer.

        Args:
            x (torch.Tensor): Input node features of shape (num_nodes, num_grid_points, in_dim).
            edge_attr_spatial (torch.Tensor): Edge features for the spatial interactions (num_edges, attr_dim).
            edge_attr_fiber (torch.Tensor): Edge features for the within fiber interactions (num_grid_points, num_grid_points, attr_dim).
            edge_index (torch.Tensor): Graph connectivity in COO format with shape (2, num_edges).
            batch_target (torch.Tensor, optional): Batch vector of target nodes of shape (num_nodes,). (default: `None`)
            batch_source (torch.Tensor, optional): Batch vector of source nodes of shape (num_nodes,). (default: `None`)

        Returns:
            out (torch.Tensor): Output node features of shape (num_nodes, num_grid_points, out_dim).
        """
        if batch_source is not None and batch_target is not None:
            size = (batch_source.shape[0], batch_target.shape[0])
        else:
            size = None
        return self.propagate(edge_index, x=x, edge_attr_spatial=edge_attr_spatial, edge_attr_fiber=edge_attr_fiber, size = size)

    def message(self, x_j, edge_attr_spatial):
        out = self.kernel_spatial(edge_attr_spatial) * x_j  # [B, S, C] * [B, S, C], S = num elements in source fiber
        return out
    
    def update(self, message_aggr, edge_attr_fiber, x):
        x_spatially_convolved = message_aggr
        kernel_fiber = self.kernel_fiber(edge_attr_fiber)
        if not self.depthwise:
            kernel_fiber = kernel_fiber.unflatten(-1, (self.out_dim, self.in_dim))
        x_fully_convolved = torch.einsum(self.einsum_eq, kernel_fiber, x_spatially_convolved) * 1.
        if self.training and not(self.callibrated):
            self.callibrate(x.std(), x_spatially_convolved.std(), x_fully_convolved.std())
        if self.bias is not None:
            return x_fully_convolved + self.bias
        else:  
            return x_fully_convolved
    
    def callibrate(self, std1, std2, std3):
        print('Callibrating...')
        with torch.no_grad():
            self.kernel_spatial.weight.data = self.kernel_spatial.weight.data * std1/std2
            self.kernel_fiber.weight.data = self.kernel_fiber.weight.data * std2/std3
            self.callibrated = ~self.callibrated
