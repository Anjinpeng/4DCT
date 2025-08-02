#损失函数
#2025.3.11 By An Jinpeng

import torch

#L1损失
def l1_loss(network_output,gt):
    return torch.abs(network_output-gt).mean()

#l2损失
def l2_loss(network_output,gt):
    return ((network_output-gt)**2).mean()


    
#TV正则化
def tv_3d_loss(vol, reduction="sum"):

    dx = torch.abs(torch.diff(vol, dim=0))
    dy = torch.abs(torch.diff(vol, dim=1))
    dz = torch.abs(torch.diff(vol, dim=2))

    tv = torch.sum(dx) + torch.sum(dy) + torch.sum(dz)

    if reduction == "mean":
        total_elements = (
            (vol.shape[0] - 1) * vol.shape[1] * vol.shape[2]
            + vol.shape[0] * (vol.shape[1] - 1) * vol.shape[2]
            + vol.shape[0] * vol.shape[1] * (vol.shape[2] - 1)
        )
        tv = tv / total_elements
    return tv












