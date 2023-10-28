import os
import numpy as np
import cv2
import torch

import matplotlib.pyplot as plt
from skimage.io import imread
from utils import *

from pytorch3d.structures import Meshes
from pytorch3d.io import IO
from pytorch3d.renderer import (
    look_at_view_transform,
    look_at_rotation,
    OpenGLPerspectiveCameras,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    AmbientLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
    BlendParams,
)

from RasterModel.rastermodel import RaseterObjectModel, UoM
from RasterModel.view_points import sample_views

def convert_bop_pose_to_p3d(RT):
    Rz = torch.tensor([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float()

    RT = torch.matmul(Rz, RT)

    R = RT[:3, :3].t().reshape(1, 3, 3)
    T = RT[:3, 3].reshape(1, 3)

    return R, T

def convert_bop_cam_to_p3d(R, T, K, w, h, b, device):
    f_x, f_y = K[0, 0], K[1, 1]
    p_x, p_y = K[0, 2], K[1, 2]
    f = torch.tensor((f_x, f_y), dtype=torch.float32).unsqueeze(0)
    p = torch.tensor((p_x, p_y), dtype=torch.float32).unsqueeze(0)
    # img_size = torch.tensor((h, w), dtype=torch.float32).unsqueeze(0)

    camera = PerspectiveCameras(
        R=R, T=T, focal_length=f, principal_point=p, image_size=((h, w),), device=device, in_ndc=False
    )
    return camera

def sample_viewpoints_p3d(n, r):
    elev = torch.linspace(0, 360, n)
    azim = torch.linspace(-180, 180, n)
    R, T = look_at_view_transform(dist = r, elev = elev, azim = azim) 
    return R, T

def sample_viewpoints_bop(n, r):
    vps, _ = sample_views(n, r, mode='fibonacci')
    R = np.zeros((n,3,3))
    T = np.zeros((n,3))

    for i in range(len(vps)):
        R[i] = vps[i]['R']
        T[i] = vps[i]['t']

    return R, T



# Params
K = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])

f_x, f_y = K[0, 0], K[1, 1]
p_x, p_y = K[0, 2], K[1, 2]
h = 480
w = 640

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Pose
vps = 16
R, T = sample_viewpoints_p3d(vps, 1)

obj_filename = 'data/lm_models/obj_000001.ply'

# pytorch 3D
device = torch.device("cuda:0")
mesh = IO().load_mesh(obj_filename).to(device)
mesh.scale_verts_(0.001)
meshes = mesh.extend(vps)
camera = convert_bop_cam_to_p3d(R,T,K,w,h,vps,device)
lights = AmbientLights(device=device)
blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
# Set Renderer Parameters
raster_settings = RasterizationSettings(
    image_size=(h, w),
    blur_radius=0.0,
    faces_per_pixel=1,
    max_faces_per_bin=mesh.faces_packed().shape[0],
    perspective_correct=True,
)

rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)

renderer = MeshRenderer(
    rasterizer,
    shader=SoftPhongShader(
        device=device,
        cameras=camera,
        lights=lights,
        blend_params=blend_params,
    ),
)

p3d_images = renderer(meshes, cameras=camera, lights=lights)
# Plot the rendered images
image_grid(p3d_images.cpu().numpy(), rows=int(np.sqrt(vps)), cols=int(np.sqrt(vps)), rgb=True)

# raster model
rasterModel = RaseterObjectModel("data/lm_models/obj_000001.ply",uom=UoM.MILLIMETER)
rasterModel.setCamParams(K, w, h)




