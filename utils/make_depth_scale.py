import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import json
from plyfile import PlyData, PlyElement
from read_write_model import *
import sys
sys.path.append(".")
from scene.dataset_readers import readNerfiesCameras2,readNerfiesCameras
from typing import NamedTuple, Optional
from utils.graphics_utils import getWorld2View2, getProjectionMatrix2


# Camera = collections.namedtuple(
#     "Camera", ["id","width", "height"]
# )
# BaseImage = collections.namedtuple(
#     "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys"]
# )

# class Image(BaseImage):
#     def qvec2rotmat(self):
#         return qvec2rotmat(self.qvec)
    
# def qvec2rotmat(qvec):
#     return np.array(
#         [
#             [
#                 1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
#                 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
#                 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
#             ],
#             [
#                 2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
#                 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
#                 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
#             ],
#             [
#                 2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
#                 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
#                 1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
#             ],
#         ]
#     )


# def rotmat2qvec(R):
#     Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
#     K = (
#         np.array(
#             [
#                 [Rxx - Ryy - Rzz, 0, 0, 0],
#                 [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
#                 [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
#                 [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
#             ]
#         )
#         / 3.0
#     )
#     eigvals, eigvecs = np.linalg.eigh(K)
#     qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
#     if qvec[0] < 0:
#         qvec *= -1
#     return qvec    
    

def get_scales(key, cameras, images, points3d_ordered, args):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    # pts_idx = images_metas[key].point3D_ids

    # mask = pts_idx >= 0
    # mask *= pts_idx < len(points3d_ordered)

    # pts_idx = pts_idx[mask]
    # valid_xys = image_meta.xys[mask]
    
    pts = points3d_ordered
    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R) + image_meta.tvec
    valid_xys = image_meta.xys
    # if len(pts_idx) > 0:
    #     pts = points3d_ordered[pts_idx]
    # else:
    #     pts = np.array([0, 0, 0])

    # R = qvec2rotmat(image_meta.qvec)
    # psts = np.dot(pts, R.T) + image_meta.tvec

    invcolmapdepth = 1. / pts[..., 2] 
    # n_remove = len(image_meta.name.split('.')[-1]) + 1
    # invmonodepthmap = cv2.imread(f"{args.depths_dir}/{image_meta.name[:-n_remove]}.png", cv2.IMREAD_UNCHANGED)
    invmonodepthmap = cv2.imread(f"{args.depths_dir}/{image_meta.name}.png", cv2.IMREAD_UNCHANGED)

    
    if invmonodepthmap is None:
        return None
    
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]

    # invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)
    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2 ** 8)
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    maps = (valid_xys * s).astype(np.float32)
    valid = (
        (maps[..., 0] >= 0) * 
        (maps[..., 1] >= 0) * 
        (maps[..., 0] < cam_intrinsic.width * s) * 
        (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))
    # print(f"valid sum {valid.sum()}")
    # print(f"maps.shape {maps.shape}")
    # print(f"maps.min {np.min(maps)}, maps.max{np.max(maps)}")
    # print(f"invmonodepthmap.shape {invmonodepthmap.shape}")
    
    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        # print(f"maps.shape {maps.shape}")
        # print(f"maps.min {np.min(maps)}, maps.max{np.max(maps)}")
        # print(f"invmonodepthmap.shape {invmonodepthmap.shape}")
        invcolmapdepth = invcolmapdepth[valid]
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]
        
        ## Median / dev
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        # print(f"t_colmap {t_colmap}, s_colmap {s_colmap}, t_momo {t_mono}, s_momo {s_mono}")
        scale = s_colmap / s_mono
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    # return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}
    return {"image_name": image_meta.name, "scale": scale, "offset": offset}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default="../data/big_gaussians/standalone_chunks/campus")
    parser.add_argument('--depths_dir', default="../data/big_gaussians/standalone_chunks/campus/depths_any")
    parser.add_argument('--model_type', default="bin")
    args = parser.parse_args()
    
    zfar = 100.0
    znear = 0.01
    ply_path = os.path.join(args.base_dir, "static_points3d.ply")
    pcd = PlyData.read(ply_path)
    
    points3d_ordered = np.array([list(vertex) for vertex in pcd['vertex']])[:, :3]
    if points3d_ordered.shape[0] > 32767:
        indices = np.random.choice(points3d_ordered.shape[0], size= 30000, replace=False)
        points3d_ordered = points3d_ordered[indices]
        print(f"point shape :{points3d_ordered.shape[0]}")
    cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras2(args.base_dir)
    # cam_infos = cam_infos[:train_num]

    images_metas = {}
    cam_intrinsics = {}
    max_var = np.max(points3d_ordered, axis=0)
    min_var = np.min(points3d_ordered, axis=0)
    # print(f"max: {max_var}, min: {min_var}")
    # print(points3d_ordered.shape[0])
    for cam_info in cam_infos:
        qvec = rotmat2qvec(cam_info.R)
        W2V = getWorld2View2(cam_info.R, cam_info.T)
        projection_matrix = getProjectionMatrix2(znear=znear, zfar=zfar, fovX=cam_info.FovX, fovY=cam_info.FovY)
        full_proj_transform = projection_matrix @ W2V
        ones = np.ones((points3d_ordered.shape[0], 1))
        points_homo = np.concatenate([points3d_ordered, ones], axis = 1)
        clip_points = points_homo @ full_proj_transform
        w = clip_points[:, 3]
        ndc_points = clip_points[:, :3]
        x_ndc = ndc_points[:, 0]
        y_ndc = ndc_points[:, 1]

        x_screen = ((x_ndc + 1) / 2.0) * cam_info.width
        y_screen = ((1 - (y_ndc + 1) / 2.0)) * cam_info.height

        screen_points = np.stack([x_screen, y_screen], axis=1)

        images_metas[cam_info.uid] = Image(id=cam_info.uid, qvec=qvec, tvec=cam_info.T, camera_id=cam_info.uid, name=cam_info.image_name, xys = screen_points, point3D_ids = None)
        cam_intrinsics[cam_info.uid] = Camera(id = cam_info.uid, model = None, width=cam_info.width, height=cam_info.height, params= None)
        
    

    # cam_intrinsics, images_metas, points3d = read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")

    # pts_indices = np.array([points3d[key].id for key in points3d])
    # pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    # points3d_ordered = np.zeros([pts_indices.max()+1, 3])
    # points3d_ordered[pts_indices] = pts_xyzs

    # depth_param_list = [get_scales(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas]
    depth_param_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(get_scales)(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas
    )

    depth_params = {
        depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
        for depth_param in depth_param_list if depth_param != None
    }

    with open(f"{args.base_dir}/depth_params.json", "w") as f:
        json.dump(depth_params, f, indent=2)

    print(0)