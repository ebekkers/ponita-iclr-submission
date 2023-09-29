import torch
from .conv import Conv
from torch.nn import LayerNorm


class ConvNext(torch.nn.Module):
    """
    A PyTorch module that implements the ConvNext layer as proposed in "A ConvNet for the 2020s"
    (CVPR 2022). See https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py 
    The layer is composed of a depthwise convolutional layer, two fully connected layers, a
    normalization layer, and a residual connection. Relative to the original implementation we do 
    not use drop path.

    Args:
        feature_dim (int): The number of input and output features.
        attr_dim (int): The number of edge features, this conditions the convolution kernel (see Conv)
        act (torch.nn.Module, optional): The activation function to use between the fully connected layers.
            Default is GELU.
        layer_scale (float, optional): Initial value for the scaling factor for the output of the layer. Default is 1e-6.
        aggr (str, optional): The aggregation method to use during convolutional message passing. Can be "add", "mean",
            or "max". Default is "add".
    """
    def __init__(self, feature_dim, attr_dim, act=torch.nn.GELU(), layer_scale=1e-6,aggr="add"): 
        super().__init__()

        self.conv = Conv(feature_dim, feature_dim, attr_dim, aggr=aggr, depthwise=True)
        self.act_fn = act
        self.linear_1 = torch.nn.Linear(feature_dim, 4 * feature_dim)
        self.linear_2 = torch.nn.Linear(4 * feature_dim, feature_dim)
        self.layer_scale = torch.nn.Parameter(torch.ones(1, feature_dim) * layer_scale)
        self.norm = LayerNorm(feature_dim)

    def forward(self, x, attr, edge_index, batch, batch_source=None):
        """
        Forward pass of the ConvNext layer.

        Args:
            x (torch.Tensor): The input tensor of shape (num_nodes, feature_dim).
            attr (torch.Tensor): The edge feature tensor of shape (num_edges, attr_dim), this conditions the convolution kernel.
            edge_index (torch.Tensor): The edge index tensor of shape (2, num_edges).
            batch (torch.Tensor): The (target) batch tensor of shape (num_nodes,) used for message passing.
            batch_source (torch.Tensor, optional): The source batch tensor of shape (num_edges,) used
                for message passing. Required when `batch` is not a unique mapping as could be in the bi-partite case. Default is None.

        Returns:
            torch.Tensor: The output tensor of shape (num_nodes, feature_dim).
        """
        input = x
        x = self.conv(x, attr, edge_index, batch_target=batch, batch_source=batch_source)
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)
        x = self.layer_scale * x
        if input.shape == x.shape: 
            x = x + input
        return x
