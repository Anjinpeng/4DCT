
import torch
import math
import numpy as np
from typing import NamedTuple



def batch_quaternion_multiply(q1, q2):
    """
    Multiply batches of quaternions.
    
    Args:
    - q1 (torch.Tensor): A tensor of shape [N, 4] representing the first batch of quaternions.
    - q2 (torch.Tensor): A tensor of shape [N, 4] representing the second batch of quaternions.
    
    Returns:
    - torch.Tensor: The resulting batch of quaternions after applying the rotation.
    """
    # Calculate the product of each quaternion in the batch
    w = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    x = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    y = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    z = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]

    # Combine into new quaternions
    q3 = torch.stack((w, x, y, z), dim=1)
    
    # Normalize the quaternions
    norm_q3 = q3 / torch.norm(q3, dim=1, keepdim=True)
    
    return norm_q3

def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)  # w2c
    return np.float32(Rt)


def getProjectionMatrix(fovX, fovY, mode, scanner_cfg):
    if mode == 0:  # Parallel beam
        # sVoxel = scanner_cfg["sVoxel"][0]  # Assume cube only!
        # sVoxel_half = sVoxel / 2
        # DSO = scanner_cfg["DSO"]
        # near = DSO - sVoxel_half
        # far = DSO + sVoxel_half
        # top = sVoxel_half
        # bottom = -sVoxel_half
        # right = sVoxel_half
        # left = -sVoxel_half
        # P = torch.zeros(4, 4)
        # z_sign = 1.0
        # P[0, 0] = 2.0 / (right - left)
        # P[0, 3] = -(right + left) / (right - left)
        # P[1, 1] = 2.0 / (top - bottom)
        # P[1, 3] = -(top + bottom) / (top - bottom)
        # P[2, 2] = z_sign * 2.0 / (far - near)
        # P[2, 3] = -(far + near) / (far - near)
        # P[3, 3] = z_sign

        # Projection matrix is eye
        P = torch.eye(4)
    elif mode == 1:
        znear = 0.01
        zfar = 100.0
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
    else:
        raise ValueError("Unsupported mode!")
    return P
