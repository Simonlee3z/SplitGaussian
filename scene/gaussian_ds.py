import torch
import numpy as np
from scene.gaussian_model import GaussianModel
from scene.color_model import Sandwich, SandwichV2
from utils.general_utils import build_rotation

class GaussianDS:
    def __init__(self, max_sh_degree: int):
        self.static_gaussian = GaussianModel(sh_degree=max_sh_degree)
        self.dynamic_gaussian = GaussianModel(sh_degree=max_sh_degree)
        self.rgbdecoder = SandwichV2(9,3)
        self.max_sh_degree = max_sh_degree
        self.color_mask = None
        
        
    #static
    @property
    def get_static_xyz(self):
        return self.static_gaussian.get_xyz
        
    @property
    def get_static_rotation(self):
        return self.static_gaussian.get_rotation
        
    @property
    def get_static_scaling(self):
        return self.static_gaussian.get_scaling
        
    @property
    def get_static_features(self):
        return self.static_gaussian.get_features
        
    @property
    def get_static_opacity(self):
        return self.static_gaussian.get_opacity
        
    def get_static_covariance(self, scaling_modifier = 1.0):
        return self.static_gaussian.get_covariance(self.get_static_scaling, scaling_modifier, self.get_static_rotation)
        
    #dynamic
    @property
    def get_dynamic_xyz(self):
        return self.dynamic_gaussian.get_xyz
        
    @property
    def get_dynamic_scaling(self):
        return self.dynamic_gaussian.get_scaling
        
    @property
    def get_dynamic_rotation(self):
        return self.dynamic_gaussian.get_rotation
        
    @property
    def get_dynamic_features(self):
        return self.dynamic_gaussian.get_features
        
    @property
    def get_dynamic_opacity(self):
        return self.dynamic_gaussian.get_opacity
        
    def get_dynamic_covariance(self, scaling_modifier = 1.0):
        return self.dynamic_gaussian.get_covariance(self.get_dynamic_scaling, scaling_modifier, self.get_dynamic_rotation)
    
    @property
    def get_xyz(self):
        return torch.cat((self.get_static_xyz, self.get_dynamic_xyz), dim = 0)
    
    @property
    def get_scaling(self):
        return torch.cat((self.get_static_scaling, self.get_dynamic_scaling), dim = 0)
    
    @property
    def get_rotation(self):
        return torch.cat((self.get_static_rotation, self.get_dynamic_rotation), dim = 0)
    
    @property
    def get_features(self):
        return torch.cat((self.get_static_features, self.get_dynamic_features), dim = 0)
    
    @property
    def get_opacity(self):
        return torch.cat((self.get_static_opacity, self.get_dynamic_opacity), dim = 0)
    
    @property
    def get_active_sh_degree(self):
        assert self.dynamic_gaussian.active_sh_degree == self.static_gaussian.active_sh_degree
        return self.dynamic_gaussian.active_sh_degree
    
    @property
    def get_max_radii2D(self):
        return torch.cat((self.static_gaussian.max_radii2D, self.dynamic_gaussian.max_radii2D), dim = 0)
    
    @property
    def get_xyz_gradient_accum(self):
        return torch.cat((self.static_gaussian.xyz_gradient_accum, self.dynamic_gaussian.xyz_gradient_accum), dim = 0)
    
    @property
    def get_denom(self):
        return torch.cat((self.static_gaussian.denom, self.dynamic_gaussian.denom), dim = 0)

    def get_covariance(self, scaling_modifier = 1.0):
        return torch.cat((self.get_static_covariance(scaling_modifier=scaling_modifier), self.get_dynamic_covariance(scaling_modifier=scaling_modifier)), dim = 0)        
        
        
    def oneupSHdegree(self):
        self.dynamic_gaussian.oneupSHdegree()
        self.static_gaussian.oneupSHdegree()
            
    def set_rgbdecoder(self):
        if self.dynamic_gaussian is None or self.static_gaussian is None:
            print("Dynamic Gaussian or Static Gaussian is not initialized.")
        self.dynamic_gaussian.set_rgbdecoder(self.rgbdecoder)
        self.static_gaussian.set_rgbdecoder(self.rgbdecoder)
            
    def get_rgbdecoder(self):
        return self.rgbdecoder    
        
    def training_setup(self, opt):
        self.static_gaussian.training_setup(opt)
        # self.static_gaussian.training_setup_static(opt)
        self.dynamic_gaussian.training_setup(opt)

    def save_rgbdecoder(self, path):
        torch.save(self.rgbdecoder.state_dict(), path)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.get_xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim = -1, keepdim= True)

        self.get_denom[update_filter] += 1

    def conversion_static_to_dynamic(self, max_grad, min_opacity, extent, max_screen_size, conversion_rate = 0.02):
        static_grads = self.static_gaussian.xyz_gradient_accum / self.static_gaussian.denom
        static_grads[static_grads.isnan()] = 0.0
        
        conversion_pts_num = int(self.static_gaussian.get_xyz.shape[0] * conversion_rate)

        top_grads, indices = torch.topk(static_grads.flatten(), conversion_pts_num)

        selected_pts_mask = torch.zeros(self.static_gaussian.get_xyz.shape[0], dtype = torch.bool, device = "cuda")
        selected_pts_mask[indices] = True

        # selected_pts_mask = torch.where(torch.norm(static_grads, dim=-1) >= max_grad, True, False)
        # selected_pts_mask_1 = torch.logical_and(selected_pts_mask, torch.max(self.static_gaussian.get_scaling, dim = 1).values <= self.static_gaussian.percent_dense*extent)
        selected_pts_mask_1 = torch.logical_and(selected_pts_mask, torch.max(self.static_gaussian.get_scaling, dim = 1).values <= 0.01 * extent)
        print(f"selected_pts_mask_nums:{selected_pts_mask_1.sum()}")
         
        new_xyz = self.static_gaussian._xyz[selected_pts_mask_1]
        new_features_dc = self.static_gaussian._features_dc[selected_pts_mask_1]
        new_features_rest = self.static_gaussian._features_rest[selected_pts_mask_1]
        new_opacities = self.static_gaussian._opacity[selected_pts_mask_1]
        new_scaling = self.static_gaussian._scaling[selected_pts_mask_1]
        new_rotation = self.static_gaussian._rotation[selected_pts_mask_1]
        
        self.dynamic_gaussian.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)
        
        # selected_pts_mask_2 = torch.logical_and(selected_pts_mask, torch.max(self.static_gaussian.get_scaling, dim = 1).values > self.static_gaussian.percent_dense * extent)
        
        # stds = self.static_gaussian.get_scaling[selected_pts_mask_2].repeat(2, 1)
        # means = torch.zeros((stds.size(0), 3), device="cuda")
        # samples = torch.normal(mean=means, std=stds)
        # rots = build_rotation(self.static_gaussian._rotation[selected_pts_mask_2]).repeat(2, 1, 1)
        # new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.static_gaussian.get_xyz[selected_pts_mask_2].repeat(2, 1)
        # new_scaling = self.static_gaussian.scaling_inverse_activation(self.static_gaussian.get_scaling[selected_pts_mask_2].repeat(2, 1) / (0.8 * 2))
        # new_rotation = self.static_gaussian._rotation[selected_pts_mask_2].repeat(2, 1)
        # new_features_dc = self.static_gaussian._features_dc[selected_pts_mask_2].repeat(2, 1, 1)
        # new_features_rest = self.static_gaussian._features_rest[selected_pts_mask_2].repeat(2, 1, 1)
        # new_opacity = self.static_gaussian._opacity[selected_pts_mask_2].repeat(2, 1)
        
        # self.dynamic_gaussian.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        # prune_mask = (self.static_gaussian.get_opacity < min_opacity).squeeze()
        # if max_screen_size:
        #     big_points_vs = self.static_gaussian.max_radii2D > max_screen_size
        #     big_points_ws = self.static_gaussian.get_scaling.max(dim = 1).values > 0.1 * extent
        #     prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        #     prune_mask = torch.logical_or(selected_pts_mask, prune_mask)     
        # self.static_gaussian.prune_points(prune_mask)
        self.static_gaussian.prune_points(selected_pts_mask_1)

        