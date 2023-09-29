import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class RFF_R3S2(nn.Module):
    def __init__(self, out_features, sigma, lmax):
        super().__init__()

        if out_features % 2 != 0:
            self.compensation = 1
        else:
            self.compensation = 0

        freqs_pos_dist = torch.randn(int(out_features / 2) + self.compensation, 1) * sigma / np.sqrt(2)
        freqs_pos_angle = torch.randint(0, lmax, (int(out_features / 2) + self.compensation, 1)).type_as(freqs_pos_dist)
        freqs_ori_angle = torch.randint(0, lmax, (int(out_features / 2) + self.compensation, 1)).type_as(freqs_pos_dist)

        self.register_buffer("freqs_pos_dist", freqs_pos_dist)
        self.register_buffer("freqs_pos_angle", freqs_pos_angle)
        self.register_buffer("freqs_ori_angle", freqs_ori_angle)

    def forward(self, x):
        x_pos_dist = F.linear(x[...,0,None], self.freqs_pos_dist)
        x_pos_angle = F.linear(x[...,1,None], self.freqs_pos_angle)
        x_ori_angle = F.linear(x[...,2,None], self.freqs_ori_angle)
        x_pos_dist = torch.cat((x_pos_dist.sin(), x_pos_dist.cos()), dim=-1)
        x_pos_angle = torch.cat((x_pos_angle.cos(), x_pos_angle.cos()), dim=-1)
        x_ori_angle = torch.cat((x_ori_angle.cos(), x_ori_angle.cos()), dim=-1)
        x = x_pos_dist * x_pos_angle * x_ori_angle
        if self.compensation:
            x = x[..., :-1]
        return x
    
class RFF_R3S2_Seperable(nn.Module):
    def __init__(self, out_features, sigma, lmax):
        super().__init__()

        if out_features % 2 != 0:
            self.compensation = 1
        else:
            self.compensation = 0

        freqs_pos_dist = torch.randn(int(out_features / 2) + self.compensation, 1) * sigma / np.sqrt(2)
        freqs_pos_angle = torch.randint(0, lmax, (int(out_features / 2) + self.compensation, 1))
        freqs_ori_angle = torch.randint(0, lmax, (int(out_features / 2) + self.compensation, 1))

        self.register_buffer("freqs_pos_dist", freqs_pos_dist)
        self.register_buffer("freqs_pos_angle", freqs_pos_angle)
        self.register_buffer("freqs_ori_angle", freqs_ori_angle)

    def forward(self, x_pos, x_ori):
        x_pos_dist = F.linear(x_pos[...,0,None], self.freqs_pos_dist)
        x_pos_angle = F.linear(x_pos[...,1,None], self.freqs_pos_angle.type_as(x_pos))
        x_ori_angle = F.linear(x_ori[...,0,None], self.freqs_ori_angle.type_as(x_pos))
        x_pos_dist = torch.cat((x_pos_dist.sin(), x_pos_dist.cos()), dim=-1)
        x_pos_angle = torch.cat((x_pos_angle.cos(), x_pos_angle.cos()), dim=-1)
        x_ori_angle = torch.cat((x_ori_angle.cos(), x_ori_angle.cos()), dim=-1)
        if self.compensation:
            return x_pos_dist[..., :-1] * x_pos_angle[..., :-1], x_ori_angle[...,:-1]
        else:
            return x_pos_dist * x_pos_angle, x_ori_angle
