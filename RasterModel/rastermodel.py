# import utils.example as d2co
# import utils.d2co.pyd2co as d2co
import RasterModel.lib.pyrastermodel as rastermodel
import cv2
import numpy as np
from plyfile import PlyData, PlyElement


UoM = rastermodel.UoM # including : METER, CENTIMETER, MILLIMETER, DECIMILLIMETER, CENTIMILLIMETER, MICRON
UoM_METER = UoM.METER
UoM_MILLIMETER = UoM.MILLIMETER


def count_points(img):
    if img.ndim == 3:
        return np.count_nonzero((img))/3
    else:
        return np.count_nonzero((img))

def compute2DBoudingBox(image):
    if image.size == 3:
        raise Exception('The dim of input image should be [h, w]')

    hs,ws=np.nonzero(image) #获取所有的非零坐标 bounding box
    rect = (0, 0, 0, 0)
    if hs.size != 0 and ws.size !=0:
        has_roi = True
        hmin,hmax=np.min(hs),np.max(hs)
        wmin,wmax=np.min(ws),np.max(ws)
        rect= (wmin, hmin, wmax-wmin, hmax-hmin)
    return rect

def convert_unique(fn_read,center_x=True,center_y=True,center_z=True):
    plydata = PlyData.read(fn_read)

    #x,y,z : embbedding to RGB
    x_ct = np.mean(plydata.elements[0].data['x'])    
    if not(center_x):
        x_ct=0
    x_abs = np.max(np.abs(plydata.elements[0].data['x']-x_ct))
    
    y_ct = np.mean(plydata.elements[0].data['y'])    
    if not(center_y):
        y_ct=0
    y_abs = np.max(np.abs(plydata.elements[0].data['y']-y_ct))    
    
    z_ct = np.mean(plydata.elements[0].data['z'])    
    if not(center_z):
        z_ct=0
    z_abs = np.max(np.abs(plydata.elements[0].data['z']-z_ct))    
    n_vert = plydata.elements[0].data['x'].shape[0]
   
    # for i in range(n_vert):
    #     r=(plydata.elements[0].data['x'][i]-x_ct)/x_abs #-1 to 1
    #     r = (r+1)/2 #0 to 2 -> 0 to 1        
    #     g=(plydata.elements[0].data['y'][i]-y_ct)/y_abs
    #     g = (g+1)/2
    #     b=(plydata.elements[0].data['z'][i]-z_ct)/z_abs
    #     b = (b+1)/2
    #     #if b> 1: b=1
    #     #if b<0: b=0
    #     plydata.elements[0].data['red'][i]=r*255
    #     plydata.elements[0].data['green'][i]=g*255
    #     plydata.elements[0].data['blue'][i]=b*255
    # plydata.write(fn_write)        
    return x_abs,y_abs,z_abs,x_ct,y_ct,z_ct

class RaseterObjectModel():
    def __init__(self, cad_filename, step = 0.001,uom = rastermodel.UoM.MILLIMETER, epsilon=20):
        K = np.array([[572.4114, 0., 325.2611],
                    [0., 573.57043,  242.04899],
                    [0., 0.,         1.]]).astype(np.float32)
        self.cam = rastermodel.PinholeCameraModel(K, 1200, 1200)
        self.img_w = self.cam.imgWidth()
        self.img_h = self.cam.imgHeight()
        
        # x_abs,y_abs,z_abs,x_ct,y_ct,z_ct = convert_unique(cad_filename)
        # self.x_abs = x_abs / 1000.0
        # self.y_abs = y_abs / 1000.0
        # self.z_abs = z_abs / 1000.0
        # self.x_ct = x_ct / 1000.0
        # self.y_ct = y_ct / 1000.0
        # self.z_ct = z_ct / 1000.0


        self.cad = rastermodel.RasterObjectModel3D()
        self.cad.setCamModel(self.cam)
        self.cad.setStepMeters(step)
        self.cad.setOrigOffset()
        self.cad.setUnitOfMeasure(uom)
        self.cad.setNormalEpsilonByDegree(epsilon)
        self.cad.setModelFile(cad_filename)
        self.cad.computeRaster()
        pass
    
    def setCamParams(self, mat, w, h):
        mat = np.array(mat, dtype=np.float32).reshape(3, 3)
        self.cam = rastermodel.PinholeCameraModel(mat, w, h)
        self.img_w = self.cam.imgWidth()
        self.img_h = self.cam.imgHeight()
        self.cad.setCamModel(self.cam)
        self.cad.createImg2GLBufferMap()
        # self.cad.computeRaster()
    
    def setModelView(self, pose):
        self.cad.setModelView(pose.astype(np.float32))

    def projectRasterPointsAndNormals(self):
        '''
        usage: call after setModelView, and return Visible Points and their Normal direction angles in 2D
        '''
        proj_pts = self.cad.projectRasterPoints()      
        proj_nls = self.cad.projectNormalDirections() #[-pi/2, pi/2]
        return proj_pts, proj_nls

    # draw circle for each proj_point in native C++ interface
    def getEdgeMap(self, img, color=(255.0, 255.0, 255.0)):
        proj_pts = self.cad.projectRasterPoints()
        return self.cad.getEdgeMap(proj_pts, img, color)

    # draw line for each proj_line in native C++ interface
    def getEdgeMapBySegments(self, img, color=(255.0, 255.0, 255.0)):
        return self.cad.getEdgeMapBySegments(img, color)
    
    def project(self, img, color=(0, 0, 0), gray =False):     
        proj_pts = np.array(self.cad.projectRasterPoints())

        for point in proj_pts:
            point = (int(point[0]), int(point[1]))
            cv2.circle(img, point, 1, color, -1)

        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def getVisible2D3DPairs(self):
        pts_3d = np.array(self.cad.getXYZCoordinateMap())  # 3D
        pts_2d = np.array(self.cad.projectRasterPoints())  # 2D
        return pts_2d, pts_3d
    
    def getXYZCoordinateMap(self, img, color=(0, 0, 0), gray =False):
        pts_3d = np.array(self.cad.getXYZCoordinateMap())  # 3D
        pts_2d = np.array(self.cad.projectRasterPoints())  # 2D

        edge_map = img.copy()
        coord_map = np.zeros((img.shape[0], img.shape[1], 3), np.float32)

        for i, (p3d, p2d) in enumerate(zip(pts_3d, pts_2d)):
            pt_2d = (int(p2d[0]), int(p2d[1]))
            pt_3d = p3d

            r=(p3d[0]-self.x_ct)/self.x_abs #-1 to 1
            r = (r+1)/2.0 #0 to 2 -> 0 to 1        
            g=(p3d[1]-self.y_ct)/self.y_abs
            g = (g+1)/2.0
            b=(p3d[2]-self.z_ct)/self.z_abs
            b = (b+1)/2.0
            
            cv2.circle(edge_map, pt_2d, 1, color, -1)
            coord_map[pt_2d[1], pt_2d[0]] = [r,g,b]
        
        if gray:
            edge_map = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)
        
        return edge_map, coord_map

    def mask(self):
        mask = self.cad.getMask()
        return mask

# if __name__ == "__main__":
    