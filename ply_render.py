import cv2
import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import IO, ply_io
from iopath.common.file_io import PathManager
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
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.utils import cameras_from_opencv_projection

from RasterModel.rastermodel import RaseterObjectModel, UoM


K = np.array([[572.4114, 0.,         325.2611],
              [0.,        573.57043,  242.04899],
              [0.,        0.,         1.]])

f_x, f_y = K[0,0], K[1,1]
p_x, p_y = K[0,2], K[1,2]
h = 480
w = 640

device = torch.device("cuda:0")

# verts, face = ply_io.load_ply("data/lm_models/obj_000001.ply")
# verts_rgb = torch.from_numpy(np.load("textures.npy"))
# textures = TexturesVertex(verts_features = verts_rgb)
# mesh = Meshes(verts.unsqueeze(0), face.unsqueeze(0), textures).to(device)

obj_filename = "data/lm_models/obj_000001.ply"

ply_mesh = ply_io.MeshPlyFormat()
mesh = ply_mesh.read(
    obj_filename,
    True,
    device,
    PathManager())

rasterModel = RaseterObjectModel(obj_filename,uom=UoM.METER)
rasterModel.setCamParams(K, w, h)


elev = (360) * torch.rand(10)
azim = (360) * torch.rand(10) - 180

# elev= torch.tensor([219.2172, 196.6848,  26.7611, 241.7745, 288.8531, 347.6933, 150.2406,
#         289.7920, 105.1258, 207.1929])
# azim = torch.tensor([-152.0129,  -87.8176,  121.8020, -127.5354,  135.0699,  103.5086,
#         -162.2670,    4.3773,   77.4665, -100.1230])
print(elev)
print(azim)
lights = AmbientLights(device=device)

for idx in range(10):
    R, T = look_at_view_transform(dist= 150, elev=elev[idx], azim=azim[idx]) #+5*torch.randn(1)

    #Uncomment to induce x-y translation
    # T[:,:2] = 2*torch.ones(list(T.size()[:-1])+[2])
    f = torch.tensor((f_x, f_y), dtype=torch.float32).unsqueeze(0)
    p = torch.tensor((p_x, p_y), dtype=torch.float32).unsqueeze(0)
    img_size= torch.tensor((h, w), dtype=torch.float32).unsqueeze(0)
    cam_k = torch.tensor(K, dtype=torch.float32).unsqueeze(0)

    # camera = PerspectiveCameras(
    #     R=R, T=T,
    #     focal_length=f,
    #     principal_point=p,
    #         image_size=((h, w),),
    #         device=device,
    #         in_ndc=False)

    
    
    camera = cameras_from_opencv_projection(
            R=R, tvec=T,
            camera_matrix= cam_k,
            image_size= img_size
        ).to(device)
        
    
    raster_settings = RasterizationSettings(
        image_size=(h,w), 
        blur_radius=0.0, 
        faces_per_pixel=100, 
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=camera,
            lights=lights,
        )
    )

    target_images = renderer(mesh, cameras=camera, lights=lights, znear=0.0, zfar = 10000.0)
    
    r, t = R[0].cpu().numpy(), T[0].cpu().numpy().reshape(3,-1)/1000.0
    pose = np.concatenate([r, t], axis=1)
    rasterModel.setModelView(pose)
    edge_map = np.zeros((480, 640, 3), np.uint8)
    edge_map = rasterModel.project(edge_map.copy(), (255, 255, 255), False)

    img = target_images[0, ..., [2,1,0]]#[ for i in range(num_views)]

    cv2.imshow('rgb', img.cpu().numpy())
    cv2.imshow('edge', edge_map)
    k = cv2.waitKey(0)
    if k == 27:
        break