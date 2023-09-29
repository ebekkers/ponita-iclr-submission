import torch
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data
from siva.geometry import edge_attr_r3, r3_edge_to_r3s2, edge_attr_r3s2, r3_edge_to_se3, edge_attr_se3
from siva.geometry.rotation import random_matrix

class LiftedData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.x.size(0)
        if key == 'edge_index_M':
            return self.x_M.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
    
    def set_radii(self, radius_1, radius_2):
        self.radius_1 = radius_1
        self.radius_2 = radius_2


class DataSE3(LiftedData):
    def lift(self, grid = None):
        if not isinstance(self.radius_1, (int, float)):
            self.radius_1 = float(self.radius_1[0])
        if not isinstance(self.radius_2, (int, float)):
            self.radius_2 = float(self.radius_2[0])

        x_Rd = self.x
        pos_Rd = self.pos
        batch_Rd = torch.zeros(self.x.shape[0], dtype=torch.int64) if self.batch is None else self.batch
        edge_index_Rd = self.edge_index

        self.edge_index_lift = grid is None
        if self.edge_index_lift:
            # Make initial graph
            edge_index_Rd = radius_graph(pos_Rd, self.radius_1, batch_Rd, loop=False, max_num_neighbors=1000)
            
            # Use edges to define a position-orientation graph
            pos_M, x_M, node_attr_M, batch_M = r3_edge_to_se3(pos_Rd, x_Rd, edge_index_Rd, batch_Rd, self.radius_1) 
            edge_attr_Rd = node_attr_M
        else:
            b = pos_Rd.shape[0]
            n = grid.shape[0]
            pos_x = pos_Rd[:, None, :].repeat(1, n, 1).flatten(0,1)  # [b*n, 3]
            pos_SO3 = grid[None, :, :, :].repeat(b, 1, 1, 1)  # [b, n, 3, 3]
            rand_SO3 = random_matrix(b).type_as(pos_SO3)  # [b, 3, 3]
            pos_SO3 = torch.einsum('bij,bnjk->bnik', rand_SO3, pos_SO3).flatten(0,1)  # [b*n, 3, 3]
            # pos_SO3 = grid[None, :, :, :].repeat(b, 1, 1, 1).flatten(0,1)  # [b, n, 3, 3]
            pos_M = torch.cat([pos_x, pos_SO3.flatten(-2,-1)], dim=-1)  # [b*n, 12]
            x_M = x_Rd[:, None, :].repeat(1, n, 1).flatten(0,1)
            batch_M = batch_Rd[:, None].repeat(1, n).flatten(0,1)
            edge_attr_Rd = node_attr_M = None
        
        # Set adjacency using radius_2
        edge_index_M = radius_graph(pos_M[...,:3], self.radius_2, batch_M, loop=True, max_num_neighbors=1000)

        # Compute invariants of pairs of points in position orientation space
        edge_attr_M = edge_attr_se3(pos_M, edge_index_M, node_attr_M)


        self.pos = pos_Rd
        self.x = x_Rd
        self.batch = batch_Rd  # New or updated
        self.edge_index = edge_index_Rd  # New or updated
        self.edge_attr = edge_attr_Rd  # New

        self.pos_M = pos_M
        self.x_M = x_M
        self.batch_M = batch_M
        self.node_attr_M = node_attr_M
        self.edge_index_M = edge_index_M
        self.edge_attr_M = edge_attr_M


class DataR3S2(LiftedData):
    def lift(self, grid):
        if not isinstance(self.radius_1, (int, float)):
            self.radius_1 = float(self.radius_1[0])
        if not isinstance(self.radius_2, (int, float)):
            self.radius_2 = float(self.radius_2[0])

        x_Rd = self.x
        pos_Rd = self.pos
        batch_Rd = torch.zeros(self.x.shape[0], dtype=torch.int64) if self.batch is None else self.batch
        edge_index_Rd = self.edge_index

        self.edge_index_lift = grid is None
        if self.edge_index_lift:
            # Make initial graph
            edge_index_Rd = radius_graph(pos_Rd, self.radius_1, batch_Rd, loop=False, max_num_neighbors=1000)
            
            # Use edges to define a position-orientation graph
            pos_M, x_M, node_attr_M, batch_M = r3_edge_to_r3s2(pos_Rd, x_Rd, edge_index_Rd, batch_Rd) 
            edge_attr_Rd = node_attr_M
        else:
            b = pos_Rd.shape[0]
            n = grid.shape[0]
            pos_x = pos_Rd[:, None, :].repeat(1, n, 1).flatten(0,1)  # [b*n, 3]
            pos_S2 = grid[None, :, :].repeat(b, 1, 1)  # [b, n, 3]
            rand_SO3 = random_matrix(b).type_as(pos_S2)  # [b, 3, 3]
            pos_S2 = torch.einsum('bij,bnj->bni', rand_SO3, pos_S2).flatten(0,1)  # [b*n, 3, 3]
            # pos_S2 = pos_S2.flatten(0,1)
            # pos_SO3 = grid[None, :, :, :].repeat(b, 1, 1, 1).flatten(0,1)  # [b, n, 3, 3]
            pos_M = torch.cat([pos_x, pos_S2], dim=-1)  # [b*n, 6]
            x_M = x_Rd[:, None, :].repeat(1, n, 1).flatten(0,1)
            batch_M = batch_Rd[:, None].repeat(1, n).flatten(0,1)
            edge_attr_Rd = node_attr_M = None
        
        # Set adjacency using radius_2
        edge_index_M = radius_graph(pos_M[...,:3], self.radius_2, batch_M, loop=True, max_num_neighbors=1000)

        # Compute invariants of pairs of points in position orientation space
        edge_attr_M = edge_attr_r3s2(pos_M, edge_index_M, node_attr_M)


        self.pos = pos_Rd
        self.x = x_Rd
        self.batch = batch_Rd  # New or updated
        self.edge_index = edge_index_Rd  # New or updated
        self.edge_attr = edge_attr_Rd  # New

        self.pos_M = pos_M
        self.x_M = x_M
        self.batch_M = batch_M
        self.node_attr_M = node_attr_M
        self.edge_index_M = edge_index_M
        self.edge_attr_M = edge_attr_M


class DataR3(LiftedData):
    def lift(self):
        if not isinstance(self.radius_1, (int, float)):
            self.radius_1 = float(self.radius_1[0])
        if not isinstance(self.radius_2, (int, float)):
            self.radius_2 = float(self.radius_2[0])

        x_M = x_Rd = self.x
        pos_M = pos_Rd = self.pos
        batch_M = batch_Rd = torch.zeros(self.x.shape[0], dtype=torch.int64) if self.batch is None else self.batch
        edge_index_M = edge_index_Rd = radius_graph(pos_Rd, self.radius_2, batch_Rd, loop=True, max_num_neighbors=1000)

        # Compute invariants of pairs of points in position orientation space
        edge_attr_Rd = edge_attr_M = edge_attr_r3(pos_M, edge_index_M)
        node_attr_M = None

        self.pos = pos_Rd
        self.x = x_Rd
        self.batch = batch_Rd  # New or updated
        self.edge_index = edge_index_Rd  # New or updated
        self.edge_attr = edge_attr_Rd  # New

        self.pos_M = pos_M
        self.x_M = x_M
        self.batch_M = batch_M
        self.node_attr_M = node_attr_M
        self.edge_index_M = edge_index_M
        self.edge_attr_M = edge_attr_M
