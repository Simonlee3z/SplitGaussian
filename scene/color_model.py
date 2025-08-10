import torch 
import numpy as np
from simple_knn._C import distCUDA2
import torch.nn as nn
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
from utils.time_utils import ShsNetwork

            
class ShsModel:
    def __init__(self, is_blender = False):
        self.shs = ShsNetwork(is_blender=is_blender).cuda()
        self.optimizer = None
        self.shs_lr_scale = 5

    def step(self, xyz, time_emb):
        return self.shs(xyz, time_emb)
    
    def train_setting(self, training_args):
        l = [
            {'params': list(self.shs.parameters()),
             'lr': training_args.feature_lr * self.shs_lr_scale,
             'name': "shs"}
        ]
        self.optimizer = torch.optim.Adam(l,lr = 0.0, eps = 1e-15)

        self.shs_scheduler_args = get_expon_lr_func(lr_init=training_args.feature_lr * self.shs_lr_scale,
                                                    lr_final=training_args.feature_lr_final,
                                                    lr_delay_mult=training_args.feature_lr_delay_mult,
                                                    max_steps=training_args.rgb_lr_max_steps)
    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "shs/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.shs.state_dict(), os.path.join(out_weights_path, 'shs.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "shs"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "shs/iteration_{}/shs.pth".format(loaded_iter))
        self.shs.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "shs":
                lr = self.shs_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr      