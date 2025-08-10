import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence, static_loss
from gaussian_renderer import render, network_gui, render_static_gs, render_dynamic_gs, render_dynamic_gs_v2, render_static_gs_v2, render_dynamic_gs_v3, render_static_gs_v3, render_ds_gaussian
import sys
from scene import Scene, GaussianModel, DeformModel, Scene2
from scene.gaussian_ds import GaussianDS
from scene.color_model import rgbdecoder, SandwichV2,RGBDecoderVRayShiftV2, ShsModel
from utils.general_utils import safe_state, get_linear_noise_func, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, normalize_depth
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
from utils.flow_utils import visualize_opacity_flow
from utils.time_utils import get_embedder
import cv2
import json
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



def train(dataset, opt, pipe, testing_iterations,saving_iterations, usedepth, usedepthReg):
    tb_writer = prepare_output_and_logger(dataset)
    # create static scene
    gaussians = GaussianDS(dataset.sh_degree)
    scene = Scene2(dataset, gaussians)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof, spatial_lr_scale= 5)

    deform.train_setting(opt)
    gaussians.training_setup(opt)
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.merge_training)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    
    # time embedder
    time_embedder, output_dim = get_embedder(1, 1)
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_Lossstatic_for_log = 0.0
    ema_Lossdynamic_for_log = 0.0
    ema_Ll1dynamic_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=40000)

    if pipe.feature_splatting:
        rgbdec = rgbdecoder(SandwichV2(9, 3))
        # rgbdec = rgbdecoder(RGBDecoderVRayShiftV2(9, 3))
        rgbdec.train_setting(opt)
    else:
        rgbdec = None    

    if pipe.shs_model:
        shs_model = ShsModel(is_blender=dataset.is_blender)
        shs_model.train_setting(opt)
    else:
        shs_model = None     

    for iteration in range(1, opt.iterations + 1):
        iter_start.record()
        
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            
        total_frames = len(viewpoint_stack)
        time_interval = 1 / total_frames
        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid   
        time = time_embedder(torch.tensor([fid],  dtype=torch.float32,device="cuda"))  

        if iteration < opt.warm_up + 1:
            d_xyz, d_scaling, d_rotation = 0.0, 0.0, 0.0

        else:
            N = gaussians.dynamic_gaussian.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)     
            
            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)

            d_xyz, d_rotation, d_scaling = deform.step(gaussians.dynamic_gaussian.get_xyz, time_input + ast_noise)
        # stage 1
        if iteration < opt.merge_training + 1:
            d_opacity = torch.zeros((1, 1), device="cuda")
            d_feat_dc = torch.zeros((1, 1, 3), device="cuda")
            d_feat_rest = torch.zeros((1, 15, 3), device="cuda")
            mask = torch.tensor(viewpoint_cam.gt_dynamic_mask, device='cuda')
            gt_image = viewpoint_cam.original_image.cuda()

            #dynamic
            render_dynamic_pkg = render_dynamic_gs_v3(viewpoint_cam, gaussians, pipe, background, d_xyz=d_xyz, d_rotation= d_rotation, d_scaling=d_scaling, is_6dof=dataset.is_6dof, time = time)

            dynamic_image, dynamic_screenspace_points, dynamic_visibility_filter, dynamic_radii, dynamic_depth = render_dynamic_pkg["render"], render_dynamic_pkg["viewspace_points"],  render_dynamic_pkg["visibility_filter"], render_dynamic_pkg["radii"], render_dynamic_pkg["depth"]

            Ll1_d = l1_loss(dynamic_image, gt_image * (1.0 - mask))
            loss_d = (1.0 - opt.lambda_dssim) * Ll1_d + opt.lambda_dssim * (1.0 - ssim(dynamic_image, gt_image* (1.0 - mask)))

            #static
            render_static_pkg = render_static_gs_v3(viewpoint_cam, gaussians, pipe, background, time = time, d_opacity=d_opacity, d_feat_dc=d_feat_dc, d_feat_rest=d_feat_rest)
            static_image, static_screenspace_points, static_visibility_filter, static_radii, static_depth = render_static_pkg["render"],render_static_pkg["viewspace_points"], render_static_pkg["visibility_filter"],render_static_pkg["radii"], render_static_pkg["depth"]

            Ll1_s = l1_loss(static_image * mask, gt_image * mask)
            loss_s = (1.0 - opt.lambda_dssim) * Ll1_s + opt.lambda_dssim * (1.0 - ssim(static_image * mask, gt_image * mask))

            loss = loss_s + loss_d 

            # Depth regularization
            Ll1depth_pure = 0.0
            if depth_l1_weight(iteration) > 0:
                invDepth = static_depth
                invDepth_d = dynamic_depth
                mono_invdepth = viewpoint_cam.depth.cuda()
                depth_mask = (viewpoint_cam.depth > 0).cuda()

                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask * mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth 
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0   
            
            Ll1 = Ll1_s + Ll1_d
            loss.backward()


            iter_end.record()
            
            with torch.no_grad():
                #Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_Ll1depth_for_log = 0.4 * Ll1depth  + 0.6 * ema_Ll1depth_for_log
                ema_Lossstatic_for_log = 0.4 * loss_s.item() + 0.6 * ema_Lossstatic_for_log
                ema_Lossdynamic_for_log = 0.4 * loss_d.item() + 0.6 * ema_Lossdynamic_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", "Static Loss": f"{ema_Lossstatic_for_log:.{7}f}", "Dynamic Loss": f"{ema_Lossdynamic_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()   
            
                gaussians.dynamic_gaussian.max_radii2D[dynamic_visibility_filter] = torch.max(gaussians.dynamic_gaussian.max_radii2D[dynamic_visibility_filter], dynamic_radii[dynamic_visibility_filter])

                gaussians.static_gaussian.max_radii2D[static_visibility_filter] = torch.max(gaussians.static_gaussian.max_radii2D[static_visibility_filter], static_radii[static_visibility_filter])


                # Log and save
                cur_psnr = training_record(dataset,tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                        testing_iterations, scene, render, (pipe, background), deform,
                                        dataset.load2gpu_on_the_fly, dataset.is_6dof, shs_model=shs_model)
                if iteration in testing_iterations:
                    if cur_psnr.item() > best_psnr:
                        best_psnr = cur_psnr.item()
                        best_iteration = iteration

                if iteration in saving_iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                    deform.save_weights(dataset.model_path, iteration) 
                    if rgbdec != None:
                        rgbdec.save(dataset.model_path, iteration)
                    if shs_model != None:
                        shs_model.save_weights(dataset.model_path, iteration)    

                if iteration < opt.densify_until_iter:
                    # dynamic
                    d_viewspace_point_tensor_densify = render_dynamic_pkg["viewspace_points"]
                    gaussians.dynamic_gaussian.add_densification_stats(d_viewspace_point_tensor_densify, dynamic_visibility_filter)    

                    # static
                    s_viewspace_point_tensor_densify = render_static_pkg["viewspace_points"]
                    gaussians.static_gaussian.add_densification_stats(s_viewspace_point_tensor_densify, static_visibility_filter) 
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                        gaussians.dynamic_gaussian.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        
                        gaussians.static_gaussian.densify_and_prune(opt.static_densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        # gaussians.static_gaussian.densify_and_prune(0.0002, 0.005, scene.cameras_extent, size_threshold)
                        
                        print(f"dynamic_gaussian:{gaussians.dynamic_gaussian.get_xyz.shape[0]}, static_gaussians:{gaussians.static_gaussian.get_xyz.shape[0]}")

                    if iteration % opt.opacity_reset_interval == 0 or(dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.dynamic_gaussian.reset_opacity()    
                        gaussians.static_gaussian.reset_opacity()   
                # VDP
                if iteration in opt.prune_iterations:   
                    visible = gaussians.static_gaussian.denom
                    visible = visible.view(-1)
                    few_visible = visible < 20
                    opacity = gaussians.static_gaussian.get_opacity        
                    opacity = opacity.view(-1)                            
                    very_opaque = opacity >= opt.prune_opacity 
                    prune_mask = torch.logical_and(very_opaque, few_visible)
                    num_to_prune = prune_mask.to('cpu').sum().item()
                    gaussians.static_gaussian.prune_points(prune_mask)  
        # stage 2
        else:
            if shs_model:
                S_N = gaussians.static_gaussian.get_xyz.shape[0]
                
                static_time_input = fid.unsqueeze(0).expand(S_N, -1)
                d_opacity, d_feat_dc, d_feat_rest = shs_model.step(gaussians.static_gaussian.get_xyz, static_time_input)
            else:
                d_opacity = torch.zeros((1, 1), device="cuda")
                d_feat_dc = torch.zeros((1, 1, 3), device="cuda")
                d_feat_rest = torch.zeros((1, 15, 3), device="cuda")
            gt_image = viewpoint_cam.original_image.cuda()
                        

            render_pkg = render_ds_gaussian(viewpoint_cam, gaussians, pipe, background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, is_6dof=dataset.is_6dof,  time = time,  d_opacity=d_opacity, d_feat_dc=d_feat_dc, d_feat_rest=d_feat_rest)
            image, visibility_filter , radii, viewspace_points, depth= render_pkg["render"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["viewspace_points"], render_pkg["depth"]
            Ll1 = l1_loss(image, gt_image)
            

            loss=(1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            loss.backward()
            iter_end.record()

            with torch.no_grad():
                #Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                gaussians.get_max_radii2D[visibility_filter] = torch.max(gaussians.get_max_radii2D[visibility_filter], radii[visibility_filter])   

                if iteration in saving_iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                    deform.save_weights(dataset.model_path, iteration)
                    if rgbdec != None:
                        rgbdec.save(dataset.model_path, iteration)
                    if shs_model != None:
                        shs_model.save_weights(dataset.model_path, iteration)  

                if iteration < opt.merge_densify_until_iter:
                    viewspace_point_tensor_densify = render_pkg["viewspace_points"]
                    gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

                    if iteration > opt.merge_densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                        gaussians.dynamic_gaussian.densify_and_prune(0.0007, 0.005, scene.cameras_extent, size_threshold)
                        print(f"dynamic_gaussian:{gaussians.dynamic_gaussian.get_xyz.shape[0]}, static_gaussians:{gaussians.static_gaussian.get_xyz.shape[0]}")
  


        if iteration < opt.iterations:
            gaussians.static_gaussian.optimizer.step()
            gaussians.static_gaussian.update_learning_rate(iteration)
            gaussians.static_gaussian.optimizer.zero_grad(set_to_none=True)
                    
                    
            gaussians.dynamic_gaussian.optimizer.step()
            gaussians.dynamic_gaussian.update_learning_rate(iteration)
            gaussians.dynamic_gaussian.optimizer.zero_grad(set_to_none=True)
                    
            deform.optimizer.step()
            deform.optimizer.zero_grad()
            deform.update_learning_rate(iteration)
            if shs_model != None and iteration > opt.merge_training:
                shs_model.optimizer.step()
                shs_model.optimizer.zero_grad(set_to_none=True)
                shs_model.update_learning_rate(iteration-opt.merge_training)
            if rgbdec != None and iteration > opt.merge_training:
                rgbdec.optimizer.step()
                rgbdec.optimizer.zero_grad(set_to_none=True)
                rgbdec.update_learning_rate(iteration)
    
    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))
    file_name = os.path.join(dataset.model_path, "record.txt")    

                       
    
def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_record(dataset,tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene:Scene, renderFunc,renderArgs, deform,load2gpu_on_the_fly, is_6dof=False, shs_model = None):
    
    test_psnr = 0.0
    
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, {'name':'train', 'cameras':[scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device = "cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    dynamic_xyz = scene.gaussians.dynamic_gaussian.get_xyz
                    time_input = fid.unsqueeze(0).expand(dynamic_xyz.shape[0], -1) 
                    d_xyz, d_rotation, d_scaling = deform.step(dynamic_xyz.detach(), time_input)
                    static_xyz = scene.gaussians.get_static_xyz
                    if shs_model != None:
                        shs_time_input = fid.unsqueeze(0).expand(static_xyz.shape[0], -1)
                        d_opacity, d_feat_dc, d_feat_rest = shs_model.step(static_xyz.detach(), shs_time_input)

                    else: 
                        d_opacity = torch.zeros((1, 1), device="cuda")
                        d_feat_dc = torch.zeros((1, 1, 3), device="cuda")
                        d_feat_rest = torch.zeros((1, 15, 3), device="cuda")

                    image = render_ds_gaussian(viewpoint, scene.gaussians, *renderArgs, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, is_6dof=is_6dof,  d_opacity=d_opacity, d_feat_dc=d_feat_dc, d_feat_rest=d_feat_rest)["render"]
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim = 0)   
                    gts =torch.cat((gts, gt_image.unsqueeze(0)), dim = 0)
                    
                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                with torch.no_grad():            
                    l1_test = l1_loss(images, gts)
                    psnr_test = psnr(images, gts).mean()

                    if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                        test_psnr = psnr_test    
                    file_name = os.path.join(dataset.model_path, "record.txt")    
                    with open(file_name, 'a') as file:
                        # file.write("\n[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, config['name'], l1_test, psnr_test))
                        file.write("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))


                    # print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))         
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))       
                    
    return test_psnr    

def safe_ssim_batched(images, gts, batch_size=8):
    ssim_vals = []
    with torch.no_grad():
        for i in range(0, images.size(0), batch_size):
            batch_img = images[i:i+batch_size]
            batch_gt = gts[i:i+batch_size]
            val = ssim(batch_img, batch_gt)

            if val.dim() == 0:
                val = val.unsqueeze(0)
            ssim_vals.append(val)
    return torch.cat(ssim_vals).mean()        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(30000, 70001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000,2_5000 ,30_000,35_000, 40000, 50_000, 60_000, 47_000, 70_000, 80_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configpath", type=str, default = "None")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
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
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    train(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, usedepth=args.usedepth, usedepthReg=args.usedepthReg)

    # All done
    print("\nTraining complete.")