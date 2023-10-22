from __future__ import annotations

import os as os
import time as time

import mcubes as mcubes
import numpy as np
import torch as torch
from rich.console import Console
from torch.utils.data import dataset
from torch.utils.tensorboard import SummaryWriter

from nerfstudio.utils.math import safe_normalize

CONSOLE = Console(width=120)

###Very basic implimentation from https://discuss.pytorch.org/t/dataloader-shuffle-same-order-with-multiple-dataset/94800
class SSANDataset(dataset.Dataset):
    def __init__(self,depth_50,depth_16,depth_84,surface_normals,ray_origins,ray_directions,ray_cam_inds):
        self.depth_50 = depth_50
        self.depth_16 = depth_16
        self.depth_84 = depth_84
        self.surface_normals = surface_normals
        self.ray_origins = ray_origins
        self.ray_directions= ray_directions
        self.ray_cam_inds = ray_cam_inds

    def __getitem__(self,index):
        _d50 = self.depth_50[index]
        _d16 = self.depth_16[index]
        _d84 = self.depth_84[index]
        _norms = self.surface_normals[index]
        _orig = self.ray_origins[index]
        _dir = self.ray_directions[index]
        _cam = self.ray_cam_inds[index]

        return _d50, _d16, _d84, _norms, _orig, _dir, _cam
    
    ### must be used after converting depth vals to 3d points
    def to_aabb_bounding_box(self,bounding_box_min,bounding_box_max):
        if self.depth_50.shape[-1] != bounding_box_min.shape[0]:
            raise ValueError("depth_50 not the right shape. make sure you are using depth_to_point before.")

        self.depth_50 -= bounding_box_min
        self.depth_50 /= (bounding_box_max-bounding_box_min)

        self.depth_16 -= bounding_box_min
        self.depth_16 /= (bounding_box_max-bounding_box_min)

        self.depth_84 -= bounding_box_min
        self.depth_84 /= (bounding_box_max-bounding_box_min)

        self.ray_origins-= bounding_box_min
        self.ray_origins /=(bounding_box_max - bounding_box_min)


    ### Converts the depth values to a 3d pos.
    def depth_to_point(self):
        self.depth_50 = self.ray_origins+(self.ray_directions*self.depth_50)
        self.depth_16 = self.ray_origins+(self.ray_directions*self.depth_16)
        self.depth_84 = self.ray_origins+(self.ray_directions*self.depth_84)
    
    ###Converts all parameters to a 2d array. Either [n,1] or [n,3] for every
    def to_2d_array(self):
        param_shape = self.depth_50.shape
        self.depth_50 = torch.reshape(self.depth_50,[param_shape[1]*param_shape[0]*param_shape[2],self.depth_50.shape[3]])
        self.depth_16 = torch.reshape(self.depth_16,[param_shape[1]*param_shape[0]*param_shape[2],self.depth_16.shape[3]])
        self.depth_84 = torch.reshape(self.depth_84,[param_shape[1]*param_shape[0]*param_shape[2],self.depth_84.shape[3]])
        
        self.surface_normals = torch.reshape(self.surface_normals,[param_shape[1]*param_shape[0]*param_shape[2],self.surface_normals.shape[3]])
        self.surface_normals = safe_normalize(self.surface_normals)
        
        self.ray_origins = torch.reshape(self.ray_origins,[param_shape[1]*param_shape[0]*param_shape[2],self.ray_origins.shape[3]])
        self.ray_directions = torch.reshape(self.ray_directions,[param_shape[1]*param_shape[0]*param_shape[2],self.ray_directions.shape[3]])
        self.ray_cam_inds = torch.reshape(self.ray_cam_inds,[param_shape[1]*param_shape[0]*param_shape[2],self.ray_cam_inds.shape[3]])

    def shuffle_data(self):
        rand_index = torch.randperm(self.depth_50.shape[0])

        self.depth_50 = self.depth_50[rand_index]
        self.depth_16 = self.depth_16[rand_index]
        self.depth_84 = self.depth_84[rand_index]
        self.surface_normals = self.surface_normals[rand_index]
        self.ray_origins = self.ray_origins[rand_index]
        self.ray_directions = self.ray_directions[rand_index]
        self.ray_cam_inds = self.ray_cam_inds[rand_index]

        print("Dataset Shuffled")
    
    ##Shuffles and reduces dataset to the desired amount of rays
    def trim_data(self, ray_limit: int):
        self.shuffle_data()
        if(self.depth_50.shape[0] > ray_limit):

            self.depth_50 = self.depth_50[:ray_limit]
            self.depth_16 = self.depth_16[:ray_limit]
            self.depth_84 = self.depth_84[:ray_limit]
            self.surface_normals = self.surface_normals[:ray_limit]
            self.ray_origins = self.ray_origins[:ray_limit]
            self.ray_directions = self.ray_directions[:ray_limit]
            self.ray_cam_inds = self.ray_cam_inds[:ray_limit]
        else:
            print("Fewer rays than limit, all rays used.")

    ##Debug function. Write depths of rays within aabb to tensorboard
    def display_aabb_depths(self,bounding_box_min: torch.Tensor, bounding_box_max: torch.Tensor, profiler: SummaryWriter):
        greater_than_min = torch.where(self.depth_50[:,:,:]>=bounding_box_min,True,False)
        less_than_max = torch.where(self.depth_50[:,:,:] <= bounding_box_max,True,False)

        valid_datums = torch.all((torch.bitwise_and(greater_than_min,less_than_max)),dim=3)
        bounded_data = torch.where(valid_datums[:,:,:,None],self.depth_50,False)
        for x in np.linspace(0,60,5,dtype=int):
            profiler.add_image("Bounded Depth/50/",bounded_data[x,:,:].cpu().numpy().T.swapaxes(1,2),global_step=x)


    def __len__(self):
        return len(self.depth_50)

