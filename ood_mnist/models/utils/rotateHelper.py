import torch
import torch.nn.functional as F
import numpy as np


def rot_img(x, theta):
    theta = torch.tensor(theta)
    x = x.to("cpu")
    rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                            [torch.sin(theta), torch.cos(theta), 0]], dtype=torch.float)
    rot_mat = rot_mat[None, ...].repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size()).type(torch.float)
    output = F.grid_sample(x, grid)

    return output

