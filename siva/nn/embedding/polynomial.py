import torch


class PolynomialFeatures(torch.nn.Module):
    def __init__(self, degree):
        super(PolynomialFeatures, self).__init__()

        self.degree = degree

    def forward(self, x):

        polynomial_list = [x]
        for it in range(1, self.degree):
            polynomial_list.append(torch.einsum('...i,...j->...ij', polynomial_list[-1], x).flatten(-2,-1))
        return torch.cat(polynomial_list, -1)
