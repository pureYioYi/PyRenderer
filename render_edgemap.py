import os
import numpy as np
import cv2
import torch

import matplotlib.pyplot as plt
from skimage.io import imread
from utils import *

from RasterModel.rastermodel import RaseterObjectModel, UoM
from RasterModel.view_points import sample_views

def concat_R_T(R, T):
    n = R.shape[0]
    R = torch.tensor(R)
    T = torch.tensor(T)

    RT = torch.zeros((n, 4, 4))
    RT[:, 3, 3] = 1
    RT[:, :3, :3] = R
    RT[:, :3, 3] = T
    return RT

def convert_bop_pose_to_p3d(RT):
    Rz = torch.tensor([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float()
    RT = torch.tensor(RT)
    RT = torch.matmul(Rz, RT)

    n = RT.shape[0]
    Rs = torch.zeros((n, 3, 3))
    Ts = torch.zeros((n, 3))

    for i in range(n):
        Rs[i] = RT[i, :3, :3].t()
        Ts[i] = RT[i, :3, 3]

    return Rs, Ts

def sample_viewpoints_bop(n, r):
    vps, _ = sample_views(n, r, mode='fibonacci')
    n = len(vps)
    R = np.zeros((n,3,3))
    T = np.zeros((n,3))

    for i in range(n):
        R[i] = vps[i]['R']
        T[i] = vps[i]['t'].reshape(3,)

    return R, T

# Params
K = np.array([[700, 0.0, 320], [0.0, 700, 240], [0.0, 0.0, 1.0]])

f_x, f_y = K[0, 0], K[1, 1]
p_x, p_y = K[0, 2], K[1, 2]
h = 480
w = 640

# Pose
vps = 16
radius = 1.0
# R, T = sample_viewpoints_p3d(vps, 0.3) #[16,3,3] and [16,3]
R, T = sample_viewpoints_bop(vps, radius)
RT = concat_R_T(R, T)
R, T = convert_bop_pose_to_p3d(RT)

# obj_filename = 'data/lm_models/obj_000001.ply'
obj_filename = 'data/gearbox/BASE.stl'

# raster model
rasterModel = RaseterObjectModel(obj_filename, uom=UoM.MILLIMETER)
rasterModel.setCamParams(K, w, h)

edge_maps = np.zeros((vps, h, w, 3), dtype=np.uint8)
img = np.zeros((480, 640, 3), np.uint8)

for i in range(vps):
    edge_map = edge_maps[i, :, :, :3]
    # edge_map = np.zeros((vps, h, w, 3))

    # r, t = R[i].cpu().numpy(), T[i].cpu().numpy().reshape(3,-1)/1000
    pose = RT[i].cpu().numpy()
    # T = np.array([[1, 0, 0, 0.0],
    #                 [0, 1, 0.99837293, 0.0],
    #                 [0.99791058, -0.05947061, 1, 0.5]]).reshape(3, -1)
    rasterModel.setModelView(pose)
    edge = rasterModel.project(img.copy(), (0, 0, 255))
    cv2.imshow('edge', edge)
    k = cv2.waitKey(0)
    if k == 27:
        exit()
    edge_maps[i] = edge


# Plot the rendered images
image_grid(edge_maps, rows=int(np.sqrt(vps)), cols=int(np.sqrt(vps)), rgb=True)

