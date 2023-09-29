from torch_geometric.transforms import BaseTransform
from .data import DataR3, DataR3S2, DataSE3
from siva.geometry.rotation import uniform_grid, uniform_grid_s2
from siva.geometry.icosahedron import icosahedron

class RdToM(BaseTransform):
    def __init__(self, M, radius_1, radius_2, lift_on_demand=False):
        super().__init__()
        self.M = M
        self.DataM = self.LiftedDataClass(M)
        self.radius_1 = radius_1
        self.radius_2 = radius_2
        self.lift_on_demand=lift_on_demand
        if self.radius_1 < 0.:
            n = round(abs(self.radius_1))
            print('making grid with', n, 'points')
            if M == 'SE3':
                self.grid = uniform_grid(n, "matrix")
            elif M == 'R3S2':
                self.grid = uniform_grid_s2(n)
            print('done making grid')
        else:
            self.grid = None

    def __call__(self, graph):
        graph = self.DataM.from_dict(graph.to_dict())
        graph.set_radii(self.radius_1, self.radius_2)
        if not self.lift_on_demand:
            if self.grid is not None:
                graph.lift(self.grid.type_as(graph.pos))
            else:
                graph.lift()
        return graph
    
    def LiftedDataClass(self, M):
        if M == 'R3':
            return DataR3
        if M == 'R3S2':
            return DataR3S2
        if M == 'SE3':
            return DataSE3