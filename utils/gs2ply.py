from plyfile import PlyData, PlyElement
import torch
import numpy as np
from scene.gaussian_model import BasicPointCloud
from utils.sh_utils import SH2RGB
import argparse

max_sh_degree = 3

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def gs2ply(load_path, save_path):
    plydata = PlyData.read(load_path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
    num_pts = xyz.shape[0]
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))    
    storePly(save_path, xyz, SH2RGB(shs) * 255)

def gs2ply2(gaussian, save_path):
    xyz = gaussian.get_xyz().cpu().numpy()
    num_pts = xyz.shape[0]
    shs = gaussian.get_shs().cpu().numpy()
    shs = SH2RGB(shs)
    pcd = BasicPointCloud(points=xyz, colors=shs, normals=np.zeros((num_pts, 3)))
    storePly(save_path, xyz, shs * 255)

