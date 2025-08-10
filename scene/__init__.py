#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.gaussian_ds import GaussianDS
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0], is_dynamic = True):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.is_dynamic = is_dynamic

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        raydict = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
            print("Found cameras_sphere.npz file, assuming DTU data set!")
            scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, "cameras_sphere.npz", "cameras_sphere.npz")
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json file, assuming Nerfies data set!")
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print("Found calibration_full.json, assuming Neu3D data set!")
            scene_info = sceneLoadTypeCallbacks["plenopticVideo"](args.source_path, args.eval, 24)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found calibration_full.json, assuming Dynamic-360 data set!")
            scene_info = sceneLoadTypeCallbacks["dynamic360"](args.source_path)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            if self.is_dynamic:
                input_path = os.path.join(self.model_path, "input_dynamic")
                os.makedirs(input_path, exist_ok=True)
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(input_path, "input.ply"), 'wb') as dest_file:
                    dest_file.write(src_file.read())
            
            else:
                input_path = os.path.join(self.model_path, "input_static")
                os.makedirs(input_path, exist_ok=True)
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(input_path, "input.ply"), 'wb') as dest_file:
                    dest_file.write(src_file.read())      
            # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
            #                                                        'wb') as dest_file:
            #     dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)
        for cam in self.train_cameras[resolution_scale]:
            if cam.image_name not in raydict and cam.rayo is not None:
                raydict[cam.image_name] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda()
                
        for cam in self.test_cameras[resolution_scale]:
            if cam.image_name not in raydict and cam.rayo is not None:
                raydict[cam.image_name] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda()
                
        for cam in self.train_cameras[resolution_scale]:
            cam.rays = raydict[cam.image_name] 

        for cam in self.test_cameras[resolution_scale]:
            cam.rays = raydict[cam.image_name]        
          

        if self.loaded_iter:
            if is_dynamic:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),"dynamic",
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.point_cloud.points))            
                # self.gaussians.load_rgbdecoder(os.path.join(self.model_path,
                #                                  "point_cloud",
                #                                  "iteration_" + str(self.loaded_iter), "dynamic"
                #                                  "point_cloud.pt"))
            else:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),"static",
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.point_cloud.points))
                # self.gaussians.load_rgbdecoder(os.path.join(self.model_path,
                #                                  "point_cloud",
                #                                  "iteration_" + str(self.loaded_iter), "static"
                #                                  "point_cloud.pt"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)


    def save(self, iteration):
        if self.is_dynamic:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}/dynamic".format(iteration))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            # self.gaussians.save_rgbdecoder(os.path.join(point_cloud_path, "point_cloud.pt"))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}/static".format(iteration))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            # self.gaussians.save_rgbdecoder(os.path.join(point_cloud_path, "point_cloud.pt"))
        # checkpoint_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration)) 
        
        # # save rgbdecoder checkpoint
        # self.gaussians.save_rgbdecoder(os.path.join(checkpoint_path, "point_cloud.pt"))   
        # point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        # self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
class Scene2:
    gaussians:GaussianDS
    def __init__(self, args: ModelParams, gaussians: GaussianDS, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        raydict = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
            print("Found cameras_sphere.npz file, assuming DTU data set!")
            scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, "cameras_sphere.npz", "cameras_sphere.npz")
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json file, assuming Nerfies data set!")
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print("Found calibration_full.json, assuming Neu3D data set!")
            scene_info = sceneLoadTypeCallbacks["plenopticVideo"](args.source_path, args.eval, 24)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found calibration_full.json, assuming Dynamic-360 data set!")
            scene_info = sceneLoadTypeCallbacks["dynamic360"](args.source_path)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            input_dynamic_path = os.path.join(self.model_path, "input_dynamic")
            os.makedirs(input_dynamic_path, exist_ok=True)
            with open(scene_info.dynamic_path, 'rb') as src_file, open(os.path.join(input_dynamic_path, "input.ply"), 'wb') as dest_file:
                    dest_file.write(src_file.read())
                    
            input_static_path = os.path.join(self.model_path, "input_static")
            os.makedirs(input_static_path, exist_ok=True)
            with open(scene_info.static_path, 'rb') as src_file, open(os.path.join(input_static_path, "input.ply"), 'wb') as dest_file:
                    dest_file.write(src_file.read())      
            # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
            #                                                        'wb') as dest_file:
            #     dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)
        for cam in self.train_cameras[resolution_scale]:
            if cam.image_name not in raydict and cam.rayo is not None:
                raydict[cam.image_name] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda()
                
        for cam in self.test_cameras[resolution_scale]:
            if cam.image_name not in raydict and cam.rayo is not None:
                raydict[cam.image_name] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda()
                
        for cam in self.train_cameras[resolution_scale]:
            cam.rays = raydict[cam.image_name] 

        for cam in self.test_cameras[resolution_scale]:
            cam.rays = raydict[cam.image_name]        
          

        if self.loaded_iter:
            self.gaussians.dynamic_gaussian.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),"dynamic",
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.dynamic_point_cloud.points))            
                # self.gaussians.load_rgbdecoder(os.path.join(self.model_path,
                #                                  "point_cloud",
                #                                  "iteration_" + str(self.loaded_iter), "dynamic"
                #                                  "point_cloud.pt"))
            self.gaussians.static_gaussian.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),"static",
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.static_point_cloud.points))
                # self.gaussians.load_rgbdecoder(os.path.join(self.model_path,
                #                                  "point_cloud",
                #                                  "iteration_" + str(self.loaded_iter), "static"
                #                                  "point_cloud.pt"))
        else:
            self.gaussians.dynamic_gaussian.create_from_pcd(scene_info.dynamic_point_cloud, self.cameras_extent)
            self.gaussians.static_gaussian.create_from_pcd(scene_info.static_point_cloud, self.cameras_extent)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}/dynamic".format(iteration))
        self.gaussians.dynamic_gaussian.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            # self.gaussians.save_rgbdecoder(os.path.join(point_cloud_path, "point_cloud.pt"))

        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}/static".format(iteration))
        self.gaussians.static_gaussian.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            # self.gaussians.save_rgbdecoder(os.path.join(point_cloud_path, "point_cloud.pt"))
        # checkpoint_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration)) 
        
        # # save rgbdecoder checkpoint
        # self.gaussians.save_rgbdecoder(os.path.join(checkpoint_path, "point_cloud.pt"))   
        # point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        # self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
