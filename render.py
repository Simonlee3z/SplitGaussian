import torch
from scene import Scene, DeformModel, Scene2
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_ds_gaussian, render_static_gs_v3, render_dynamic_gs_v3
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, GaussianDS
import imageio
import numpy as np
import time
from scene.color_model import rgbdecoder, SandwichV2, RGBDecoderVRayShiftV2, ShsModel
from utils.time_utils import get_embedder
import json

def render_set(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians : GaussianDS, pipeline, background, deform, time_embedder,rgb, shs_model):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    t_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid
        rgb_time = time_embedder(torch.tensor([fid],  dtype=torch.float32,device="cuda"))
        xyz = gaussians.get_dynamic_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        if shs_model != None:
            # t_time_input = fid.unsqueeze(0).expand(gaussians.dynamic_gaussian.get_xyz.shape[0], -1)
            # d_shs = shs_model.step(gaussians.dynamic_gaussian.get_xyz.detach(), t_time_input)
            t_time_input = fid.unsqueeze(0).expand(gaussians.get_static_xyz.shape[0], -1)
            d_opacity, d_feat_dc, d_feat_rest = shs_model.step(gaussians.get_static_xyz.detach(), t_time_input)
        else:
            d_shs = 0.0
            d_opacity = torch.zeros((1, 1), device="cuda")
            d_feat_dc = torch.zeros((1, 1, 3), device="cuda")
            d_feat_rest = torch.zeros((1, 15, 3), device="cuda")

        # results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        results = render_ds_gaussian(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, time=rgb_time, rgb=rgb, d_opacity=d_opacity, d_feat_dc=d_feat_dc, d_feat_rest = d_feat_rest)
        rendering = results["render"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        # print(f"depth shape {depth.shape}")

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        fid = view.fid
        rgb_time = time_embedder(torch.tensor([fid],  dtype=torch.float32,device="cuda"))
        xyz = gaussians.get_dynamic_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

        torch.cuda.synchronize()
        t_start = time.time()

        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        if shs_model != None:
            # t_time_input = fid.unsqueeze(0).expand(gaussians.dynamic_gaussian.get_xyz.shape[0], -1)
            # d_shs = shs_model.step(gaussians.dynamic_gaussian.get_xyz.detach(), t_time_input)
            t_time_input = fid.unsqueeze(0).expand(gaussians.get_static_xyz.shape[0], -1)
            d_opacity, d_feat_dc, d_feat_rest = shs_model.step(gaussians.get_static_xyz.detach(), t_time_input)
        else:
            d_shs = 0.0
            d_opacity = torch.zeros((1, 1), device="cuda")
            d_feat_dc = torch.zeros((1, 1, 3), device="cuda")
            d_feat_rest = torch.zeros((1, 15, 3), device="cuda")
        # results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        results = render_ds_gaussian(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, time=rgb_time, rgb=rgb, d_opacity=d_opacity, d_feat_dc=d_feat_dc, d_feat_rest = d_feat_rest)
        

        torch.cuda.synchronize()
        t_end = time.time()
        t_list.append(t_end - t_start)

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m, Num. of GS: {xyz.shape[0]}')  

def render_static(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians:GaussianDS, pipeline, background, deform,  time_embedder,rgb, shs_model=None):
    render_path = os.path.join(model_path, name, "static_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "static_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "static_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    t_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid
        if shs_model != None:
            t_time_input = fid.unsqueeze(0).expand(gaussians.get_static_xyz.shape[0], -1)
            d_opacity, d_feat_dc, d_feat_rest = shs_model.step(gaussians.get_static_xyz.detach(), t_time_input)
        else:
            d_opacity = torch.zeros((1, 1), device="cuda")
            d_feat_dc = torch.zeros((1, 1, 3), device="cuda")
            d_feat_rest = torch.zeros((1, 15, 3), device="cuda")
        results = render_static_gs_v3(view, gaussians, pipeline, background,d_opacity=d_opacity, d_feat_dc=d_feat_dc, d_feat_rest=d_feat_rest)
        rendering = results["render"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        xyz = gaussians.get_static_xyz
        torch.cuda.synchronize()
        fid = view.fid
        t_start = time.time()
        if shs_model != None:
            t_time_input = fid.unsqueeze(0).expand(gaussians.get_static_xyz.shape[0], -1)
            d_opacity, d_feat_dc, d_feat_rest = shs_model.step(gaussians.get_static_xyz.detach(), t_time_input)
        else:
            d_shs = 0.0
            d_opacity = torch.zeros((1, 1), device="cuda")
            d_feat_dc = torch.zeros((1, 1, 3), device="cuda")
            d_feat_rest = torch.zeros((1, 15, 3), device="cuda")

        results = render_static_gs_v3(view, gaussians, pipeline, background, d_opacity=d_opacity, d_feat_dc=d_feat_dc, d_feat_rest=d_feat_rest)

        torch.cuda.synchronize()
        t_end = time.time()
        t_list.append(t_end - t_start)

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m, Num. of GS: {xyz.shape[0]}')    

def render_dynamic(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians : GaussianDS, pipeline, background, deform, time_embedder,rgb, shs_model = None):
    render_path = os.path.join(model_path, name, "dynamic_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "dynamic_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "dynamic_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    t_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid
        xyz = gaussians.get_dynamic_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render_dynamic_gs_v3(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        fid = view.fid
        xyz = gaussians.get_dynamic_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

        torch.cuda.synchronize()
        t_start = time.time()

        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        # results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        results = render_dynamic_gs_v3(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        

        torch.cuda.synchronize()
        t_end = time.time()
        t_list.append(t_end - t_start)

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m, Num. of GS: {xyz.shape[0]}')      

def interpolate_time(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians : GaussianDS, pipeline, background, deform, time_embedder, rgb, shs_model):
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 450
    idx = torch.randint(0, len(views), (1,)).item()
    view = views[0]
    renderings = []
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        rgb_time = time_embedder(torch.tensor([fid],  dtype=torch.float32,device="cuda"))
        xyz = gaussians.get_dynamic_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        if shs_model != None:
            # t_time_input = fid.unsqueeze(0).expand(gaussians.dynamic_gaussian.get_xyz.shape[0], -1)
            # d_shs = shs_model.step(gaussians.dynamic_gaussian.get_xyz.detach(), t_time_input)
            t_time_input = fid.unsqueeze(0).expand(gaussians.get_static_xyz.shape[0], -1)
            d_opacity, d_feat_dc, d_feat_rest = shs_model.step(gaussians.get_static_xyz.detach(), t_time_input)
        else:
            d_shs = 0.0
            d_opacity = torch.zeros((1, 1), device="cuda")
            d_feat_dc = torch.zeros((1, 1, 3), device="cuda")
            d_feat_rest = torch.zeros((1, 15, 3), device="cuda")
        results = render_ds_gaussian(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, time=rgb_time, rgb=rgb, d_opacity=d_opacity, d_feat_dc=d_feat_dc, d_feat_rest = d_feat_rest)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(t) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8) 

def interpolate_view(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians : GaussianDS, pipeline, background, timer, time_embedder, rgb, shs_model):
    render_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)

    frame = 150
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    render_poses = torch.stack(render_wander_path(view), 0)
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
    #                            0)

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = view.fid
        rgb_time = time_embedder(torch.tensor([fid],  dtype=torch.float32,device="cuda"))

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_dynamic_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)
        results = render_ds_gaussian(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, time=rgb_time, rgb=rgb)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        # acc = results["acc"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))
        # torchvision.utils.save_image(acc, os.path.join(acc_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)

def interpolate_all(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians:GaussianDS, pipeline, background, deform, time_embedder, rgb, shs_model):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 150
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()
        rgb_time = time_embedder(torch.tensor([fid],  dtype=torch.float32,device="cuda"))

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_dynamic_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render_ds_gaussian(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, time=rgb_time, rgb=rgb)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)           

def interpolate_poses(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians : GaussianDS, pipeline, background, timer, time_embedder, rgb, shs_model):
    render_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)
    frame = 520
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view_begin = views[0]  # Choose a specific time for rendering
    view_end = views[-1]
    view = views[idx]

    R_begin = view_begin.R
    R_end = view_end.R
    t_begin = view_begin.T
    t_end = view_end.T

    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = view.fid
        rgb_time = time_embedder(torch.tensor([fid],  dtype=torch.float32,device="cuda"))


        ratio = i / (frame - 1)

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_dynamic_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)

        results = render_ds_gaussian(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, time=rgb_time, rgb=rgb)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)

def interpolate_view_original(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians:GaussianDS, pipeline, background,
                              timer, time_embedder, rgb, shs_model=None):
    render_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 1000
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    R = []
    T = []
    for view in views:
        R.append(view.R)
        T.append(view.T)

    view = views[0]
    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = torch.Tensor([i / (frame - 1)]).cuda()
        rgb_time = time_embedder(torch.tensor([fid],  dtype=torch.float32,device="cuda"))
        if shs_model != None:
            # t_time_input = fid.unsqueeze(0).expand(gaussians.dynamic_gaussian.get_xyz.shape[0], -1)
            # d_shs = shs_model.step(gaussians.dynamic_gaussian.get_xyz.detach(), t_time_input)
            t_time_input = fid.unsqueeze(0).expand(gaussians.get_static_xyz.shape[0], -1)
            d_opacity, d_feat_dc, d_feat_rest = shs_model.step(gaussians.get_static_xyz.detach(), t_time_input)
        else:
            d_shs = 0.0
            d_opacity = torch.zeros((1, 1), device="cuda")
            d_feat_dc = torch.zeros((1, 1, 3), device="cuda")
            d_feat_rest = torch.zeros((1, 15, 3), device="cuda")

        query_idx = i / frame * len(views)
        begin_idx = int(np.floor(query_idx))
        end_idx = int(np.ceil(query_idx))
        if end_idx == len(views):
            break
        view_begin = views[begin_idx]
        view_end = views[end_idx]
        R_begin = view_begin.R
        R_end = view_end.R
        t_begin = view_begin.T
        t_end = view_end.T

        ratio = query_idx - begin_idx

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_dynamic_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)

        results = render_ds_gaussian(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, time=rgb_time, rgb=rgb, d_opacity=d_opacity, d_feat_dc=d_feat_dc, d_feat_rest = d_feat_rest)
        rendering = results["render"]
        # print(f"rendering shape{rendering.shape}")
        renderings.append(to8b(rendering.cpu().numpy()))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))

        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
    # print(f"renderings shape: {renderings.shape}")
    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)    

def render_sets(dataset: ModelParams, iteration:int, pipeline:PipelineParams, skip_train:bool, skip_test:bool, mode: str):
    with torch.no_grad():
        gaussians = GaussianDS(dataset.sh_degree)
        # sceneS = Scene(dataset, gaussians.static_gaussian, load_iteration = iteration, shuffle = False, is_dynamic= False)
        # sceneD = Scene(dataset, gaussians.dynamic_gaussian, load_iteration = iteration, shuffle = False, is_dynamic= True)
        scene = Scene2(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)
        if pipeline.feature_splatting == True:
            rgb = rgbdecoder(SandwichV2(9,3))
            # rgb = rgbdecoder(RGBDecoderVRayShiftV2(9,3))

            rgb.load(dataset.model_path)
        else:
            rgb = None    

        if pipeline.shs_model == True:
            shs_model = ShsModel(is_blender=dataset.is_blender)   
            shs_model.load_weights(dataset.model_path)    
        else:
            shs_model = None    
        time_embedder, output_dim = get_embedder(1, 1)

        bg_color = [1, 1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype = torch.float32, device = "cuda")
        
        if mode == "render":
            render_func = render_set
        elif mode == "static":
            render_func = render_static    
        elif mode == "dynamic":
            render_func = render_dynamic
        elif mode == "time":
            render_func = interpolate_time
        elif mode == "view":
            render_func = interpolate_view
        elif mode == "pose":
            render_func = interpolate_poses
        elif mode == "original":
            render_func = interpolate_view_original
        else:
            render_func = interpolate_all    

        if not skip_train:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, deform, time_embedder, rgb, shs_model)

        if not skip_test:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, deform, time_embedder, rgb, shs_model)   
            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original', 'static', 'dynamic'])
    parser.add_argument("--configpath", type=str, default = "None")
    args = get_combined_args(parser)
    defaults = vars(parser.parse_args([]))
    if os.path.exists(args.configpath) and args.configpath != "None":
        print("Overriding from config:", args.configpath)
        with open(args.configpath) as f:
            config = json.load(f)

        for k, v in config.items():
            if hasattr(args, k):
                current_val = getattr(args, k)
                default_val = defaults.get(k)

                if current_val == default_val:
                    setattr(args, k, v)
                    
                else:
                    print(f"Kept CLI override for '{k}': {current_val}")
            else:
                print(f"Unknown config key '{k}', skipping.")

        print("Finished loading config.")
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode)             