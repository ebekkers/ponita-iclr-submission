import torch
import torch.nn as nn

from torch_geometric.nn import global_mean_pool, global_add_pool

from poni.layers.embedding import MLP as MLPBasis, RFFNet
# from poni.layers.embedding import MLP3 as MLPBasis
# from poni.layers.embedding import RF as MLPBasis
# from poni.transforms import ToPONIGraph
from poni.transforms.geometric_simplicial_graph import ToPONIGraph
from poni.layers.conv import ConvBlockNeXt as ConvBlock


class PONI(nn.Module):
    """ Steerable E(3) equivariant (non-linear) convolutional network """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 output_channels,
                 num_layers,
                 norm="batch",
                 droprate=0.0,
                 pool='sum',
                 task='graph',
                 o_grid='0',
                 radius=0.,
                 conv_depth=1,
                 cond_method='weak',
                 cond_depth=1,
                 use_x_i=False,
                 embedding='identity',
                 sigma=0.2):
        super().__init__()

        # ############################ Two cases, stay on R3 or lift to R3xS2 ##########################################

        lifted_feature = 'pair'
        self.level = 0 if o_grid == '0' else 1
        self.transform = ToPONIGraph(level = self.level, lifted_feature = lifted_feature)
        # self.transform = ToPONIGraph(radius=radius, grid=o_grid, lifted_feature=lifted_feature)

        if self.level == 0:
            # node_attr_dim = self.transform.node_attr_dim_dict['0']
            # edge_attr_dim = self.transform.edge_attr_dim_dict['0_0']
            edge_attr_dim = 1
            in_channels = in_channels
        elif self.level == 1:
            # node_attr_dim = self.transform.node_attr_dim_dict['1']
            # edge_attr_dim = self.transform.edge_attr_dim_dict['1_1']
            edge_attr_dim = 4
            in_channels = in_channels * 2 if lifted_feature == 'pair' else in_channels

        # ################################### Settings for the graph NN ################################################

        self.task = task
        act_fn = torch.nn.SiLU()

        # ################### Settings for pre-embedding the geometric conditioning vector. ############################

        if not embedding == 'identity':
            self.calibrated = False
            edge_embedding_dim = hidden_channels
            # self.edge_embedding_fn = MLPBasis(edge_attr_dim, depth=3, width=hidden_channels, act_fn=torch.nn.SiLU(), size=edge_embedding_dim)
            self.edge_embedding_fn = RFFNet(edge_attr_dim, edge_embedding_dim, [hidden_channels, hidden_channels], sigma=sigma)
        else:
            edge_embedding_dim = edge_attr_dim
            self.edge_embedding_fn = torch.nn.Identity()


        # ##################################### The graph NN layers ####################################################

        # Initial node embedding layer
        self.embedding_layer = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                                             act_fn,
                                             nn.Linear(hidden_channels, hidden_channels))

        # Message passing layers
        layers = []
        for i in range(num_layers):
            layers.append(
                ConvBlock(hidden_channels, hidden_channels, edge_embedding_dim, hidden_features=hidden_channels,
                          layers=conv_depth, act_fn=torch.nn.SiLU(), cond_method=cond_method, cond_depth=cond_depth,
                          use_x_i=use_x_i, aggr="mean", norm=norm, droprate=droprate))
        self.layers = nn.ModuleList(layers)

        # Readout layers
        if task == 'graph':
            self.pre_pool = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                          act_fn,
                                          nn.Linear(hidden_channels, hidden_channels))
            self.post_pool = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                           act_fn,
                                           nn.Linear(hidden_channels, output_channels))
            self.init_pooler(pool)
        elif task == 'node':
            self.pre_pool = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                          act_fn,
                                          nn.Linear(hidden_channels, output_channels))

    def init_pooler(self, pool):
        if pool == "avg":
            self.pooler = global_mean_pool
        elif pool == "sum":
            self.pooler = global_add_pool

    def forward(self, graph):

        graph = self.transform(graph)

        # Unpack the graph
        x = graph.x_dict['{}'.format(self.level)]
        batch = graph.batch_dict['{}'.format(self.level)]
        edge_index = graph.edge_index_dict['{0}_{0}'.format(self.level)]
        edge_attr = graph.edge_attr_dict['{0}_{0}'.format(self.level)]

        # x = graph.x
        # batch = graph.batch
        # edge_index = graph.edge_index
        # edge_attr = graph.edge_attr

        # Embed the conditioning vectors
        # node_embedded = self.node_embedding_fn(pos_attr)
        if not self.calibrated:
            print('\n Calibrating embedding function!')
            self.edge_embedding_fn.calibrate(edge_attr)
            self.calibrated = True
        edge_embedded = self.edge_embedding_fn(edge_attr)

        # Embed
        x = self.embedding_layer(x)

        # Pass messages
        # for edge_embedding_fn, layer in zip(self.edge_embed_layers, self.layers):
        # print('------')
        for layer in self.layers:
            # edge_embedded = edge_embedding_fn(edge_attr)
            x = layer(x, edge_index, edge_embedded, batch)
            # print(torch.std(x.view(-1),dim=0))

        # Pre pool
        x = self.pre_pool(x)

        if self.task == 'graph':
            # Pool over nodes
            x = self.pooler(x, batch)
            # Predict
            x = self.post_pool(x)

        # Return result
        return x
