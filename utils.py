# Copyright 2024 Kaining Huang and Tianyi Zhang. All rights reserved.

from typing import Tuple, Optional
import torch
from torch import Tensor
import cv2 as cv
import numpy as np

# Experimental
def r_vec_to_mat(r_vec: Tensor)-> Tensor:
    '''
    Args:
        r_vec: Rotation vector

    Return:
        r_mat: Rotation matrix
    '''
    theta = torch.sqrt(torch.dot(r_vec, r_vec)+1e-10)
    r_vec = r_vec / theta
    skew_symmetric = torch.tensor([
        [0, -r_vec[2], r_vec[1]],
        [r_vec[2], 0, -r_vec[0]],
        [-r_vec[1], r_vec[0], 0]])
    r_mat = torch.eye(3) + torch.sin(theta) * skew_symmetric + (1 - torch.cos(theta)) * torch.matmul(skew_symmetric, skew_symmetric)
    return r_mat



def find_mask(points: Tensor,
              image_shape: Tuple[int, int],
              visualize: Optional[bool] = False,
              scale:Optional[float] = 0.85) -> Tensor:
    '''
    Input: points: N*4*2 in clock wise order, [x, y] x is horizon right, y is vertical down, origin is top left of the image
    '''

    if points.dim() != 3:
        raise ValueError("The input size of points should be N*4*2")
    
    if points.shape[1] != 4:
        raise ValueError("Four points are required")
    
    if len(image_shape) != 2:
        print(len(image_shape))
        raise ValueError("Image shape must be 2d, in i and j direction")
    
    # Shrink the points towards the center of the whole calibration target
    a1 = points[:, 2, 1]-points[:, 0, 1]
    b1 = points[:, 2, 0]-points[:, 0, 0]
    c1 = points[:, 2, 0]*points[:, 0, 1]-points[:, 0, 0]*points[:, 2, 1]

    a2 = points[:, 1, 1]-points[:, 3, 1]
    b2 = points[:, 1, 0]-points[:, 3, 0]
    c2 = points[:, 1, 0]*points[:, 3, 1]-points[:, 3, 0]*points[:, 1, 1]

    # print([(b1*c2-b2*c1)/(a1*b2-a2*b1), -(c1*a2-c2*a1)/(a1*b2-a2*b1)])
    target_center = torch.stack([(b1*c2-b2*c1)/(a1*b2-a2*b1), -(c1*a2-c2*a1)/(a1*b2-a2*b1)]).transpose(0,1).unsqueeze(1)
    # print(target_center.shape)

    points_rel_to_center = points-target_center
    new_points = points_rel_to_center*scale+target_center

    grid_y, grid_x = torch.meshgrid(torch.arange(image_shape[0]), torch.arange(image_shape[1]), indexing='ij')
    grid_x = grid_x+0.5
    grid_y = grid_y+0.5

    mask = torch.ones([new_points.shape[0], image_shape[0], image_shape[1]], dtype=float)

    for i in range(4):
        p1 = new_points[:,i,:].unsqueeze(1)
        p2 = new_points[:,(i+1)%4,:].unsqueeze(1)
        xcv = grid_x.unsqueeze(0)-p1[...,0:1]
        # Point-in-polygon test using cross product (>0 means outside)
        mask = mask * ((grid_x.unsqueeze(0) - p1[...,0:1]) * (p2[...,1:2] - p1[...,1:2]) - (grid_y.unsqueeze(0) - p1[...,1:2]) * (p2[...,0:1] - p1[...,0:1]) < 0).float()
        if visualize:
            for i in range(3):
                cv.imshow("test1", mask[0].float().numpy())
                cv.waitKey(0)
    return mask.bool()