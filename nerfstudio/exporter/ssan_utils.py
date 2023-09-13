"""
SSAN utils.
"""

# pylint: disable=no-member

from __future__ import annotations

import os as os
import time as time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import GPUtil
import matplotlib.pyplot as plt
import mcubes as mcubes
import numpy as np
import pymeshlab
import skimage.io as skio
import tinycudann as tcnn
import torch as torch
import torch.nn.functional as F
import torch.utils.data as torch_data
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rich.console import Console
from skimage import measure as skmeasure
from torch.utils.data import DataLoader, SubsetRandomSampler, dataset
from torch.utils.tensorboard import SummaryWriter
from torchtyping import TensorType

import nerfstudio.exporter.marching_cubes_utils as mcUtils
import nerfstudio.fields.nerfacto_field
from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.exporter.exporter_utils import (
    Mesh,
    render_trajectory,
    render_trajectory_tri_tsdf,
)
from nerfstudio.exporter.object_renderer import render_mesh
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.nerfacto import NerfactoModelTriDepth
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.math import safe_normalize

CONSOLE = Console(width=120)

@dataclass
class TSDFfromSSAN:
    """
    Class for creating TSDFs.
    """

    voxel_coords: TensorType[3, "xdim", "ydim", "zdim"]
    """Coordinates of each voxel in the TSDF."""
    values: TensorType["xdim", "ydim", "zdim"]
    """TSDF values for each voxel."""
    weights: TensorType["xdim", "ydim", "zdim"]
    """TSDF weights for each voxel."""

    ###Remove if TriTSDF is made into its own class
    normal_values: TensorType["xdim", "ydim", "zdim"]
    """Values of each voxel of the tsdf"""
    normal_weights: TensorType["xdim", "ydim", "zdim"]
    """weights of Normal for each voxel in TSDF"""
    #####
    
    ###
    ##taken from nerfacto field parameters##
    surface_mlp: torch.nn.Sequential
    optimiser: torch.optim.Optimizer

    colors: TensorType["xdim", "ydim", "zdim", 3]
    """TSDF colors for each voxel."""
    voxel_size: TensorType[3]
    """Size of each voxel in the TSDF. [x, y, z] size."""
    origin: TensorType[3]
    """Origin of the TSDF [xmin, ymin, zmin]."""
    truncation_margin: float = 5.0
    """Margin for truncation."""

    def to(self, device: str):
        """Move the tensors to the specified device.

        Args:
            device: The device to move the tensors to. E.g., "cuda:0" or "cpu".
        """
        self.voxel_coords = self.voxel_coords.to(device)
        self.values = self.values.to(device)
        self.weights = self.weights.to(device)
        self.normal_values = self.normal_values.to(device)
        self.normal_weights = self.normal_weights.to(device)
        self.colors = self.colors.to(device)
        self.voxel_size = self.voxel_size.to(device)
        self.origin = self.origin.to(device)
        return self

    @property
    def device(self):
        """Returns the device that voxel_coords is on."""
        return self.voxel_coords.device

    @property
    def truncation(self):
        """Returns the truncation distance."""
        # TODO: clean this up
        truncation = self.voxel_size[0] * self.truncation_margin
        return truncation

    @staticmethod
    def from_aabb(aabb: TensorType[2, 3], volume_dims: TensorType[3]):
        """Returns an instance of TSDF from an axis-aligned bounding box and volume dimensions.

        Args:
            aabb: The axis-aligned bounding box with shape [[xmin, ymin, zmin], [xmax, ymax, zmax]].
            volume_dims: The volume dimensions with shape [xdim, ydim, zdim].
        """

        origin = aabb[0]
        voxel_size = (aabb[1] - aabb[0]) / volume_dims

        # create the voxel coordinates
        xdim = torch.arange(volume_dims[0])
        ydim = torch.arange(volume_dims[1])
        zdim = torch.arange(volume_dims[2])
        grid = torch.stack(torch.meshgrid([xdim, ydim, zdim], indexing="ij"), dim=0)
        voxel_coords = origin.view(3, 1, 1, 1) + (grid * voxel_size.view(3, 1, 1, 1))

        print(voxel_coords.shape)

        # initialize the values and weights
        values = -torch.ones(volume_dims.tolist())
        normal_values = torch.zeros(volume_dims.tolist()+[3])
        normal_weights = torch.zeros(volume_dims.tolist()+[3])
        weights = torch.zeros(volume_dims.tolist())
        colors = torch.zeros(volume_dims.tolist() + [3])

        # TODO: set this properly based on the aabb From instant_ngp field
        per_level_scale = 1.4472692012786865

        growth_factor = np.exp((np.log(2048)-np.log(16)/(15-1)))
        
        #taken from nerfacto field parameters##
        # surface_mlp = tcnn.NetworkWithInputEncoding(
        #     n_input_dims=3,
        #     n_output_dims=4,
        #     encoding_config={
        #         "otype": "HashGrid",
        #         "n_levels": 15,
        #         "n_features_per_level": 2,
        #         "log2_hashmap_size": 19,
        #         "base_resolution": 16,
        #         "per_level_scale": per_level_scale,

        #     },
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 2,
        #         "seed": 210799
        #     },
        # )
        # print(surface_mlp)
        encoding = tcnn.Encoding(3,encoding_config={
                "otype": "HashGrid",
                "n_levels": 15,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,

            })
        
        network = tcnn.Network(encoding.n_output_dims, n_output_dims=4,
            network_config={"otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
                
            })
        
        surface_mlp = torch.nn.Sequential(encoding,network)
        print(surface_mlp)
        surface_mlp = surface_mlp.float()

        optimiser = torch.optim.Adam(surface_mlp.parameters(), lr=0.0001, betas=(0.9,0.99),eps=10e-15,weight_decay=10e-6)

        # TODO: move to device

        return TSDFfromSSAN(voxel_coords, values, weights,normal_values,normal_weights, surface_mlp,optimiser, colors, voxel_size, origin)
    
    @torch.no_grad()
    def get_mesh(self, output_dir: Path,profiler:SummaryWriter) -> Mesh:
        """Extracts a mesh using marching cubes."""
        #device = self.values.device
        # sample_density = 256
        ##X,Y,Z = np.linspace(-1,1,200)
        X = np.linspace(0,1,self.voxel_coords.shape[1])
        Y = np.linspace(0,1,self.voxel_coords.shape[1])
        Z = np.linspace(0,1,self.voxel_coords.shape[1])

        xx, yy, zz = np.meshgrid(X,Y,Z)
        print(xx[0,0,:],yy[0,0,:],zz[0,0,:])

        # grid = torch.zeros((self.voxel_coords.shape[1],self.voxel_coords.shape[2],self.voxel_coords.shape[3],3))
        # grid[:,:,:,0] = torch.Tensor(xx)
        # grid[:,:,:,1] = torch.Tensor(yy)
        # grid[:,:,:,2] = torch.Tensor(zz)

        ##results = -torch.ones(self.voxel_coords.shape[1],self.voxel_coords.shape[2],self.voxel_coords.shape[3])
        ##grid = grid.reshape(-1,3)


        ## calculate all z coords for an x,y plane.
        _x = 0
        _y=0
        while _x < self.voxel_coords.shape[1]:
            while _y <self.voxel_coords.shape[2]:
                # z and y switch to output in correct orientation
                self.values[_x,_y,:] = self.surface_mlp((self.voxel_coords[:,_x,:,_y]).t())[:,0]
                _y+=1
            _x+=1
            _y=0
        
        # results = self.surface_mlp(grid.to(self.device))
        # surface_value = results[:,0]
        # ##surface_value = surface_value.reshape(sample_density,sample_density,sample_density)
        # surface_value = torch.squeeze(surface_value)
        # # run marching cubes on CPU
        tsdf_values_np = self.values.cpu()
        tsdf_values_np = np.array(tsdf_values_np).astype(dtype=float)

        ##tsdf_values_np = 1 - np.abs(tsdf_values_np)
        print(f"tsdf value np: {tsdf_values_np.shape}")
        arr = np.linspace(-0.0,-0.1,10)##[-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5]
        #arr = [-0.075]
        try:
            os.mkdir(f"{output_dir}")
        except:
            print("directory Exists")
        os.chdir(output_dir)

        vertices,faces,normals,triangles = 0,0,0,0

        for x in arr:
            try:
                ##vertices,faces,normals,values = skmeasure.marching_cubes(tsdf_values_np,x,allow_degenerate=False)
                vertices, triangles = mcubes.marching_cubes(tsdf_values_np,x)
            except:
                print(f"x is not able to thresholded the marching cubes")

            # adjust voxel size to x,z,y as above
            voxel_size = self.voxel_size.cpu() #* torch.tensor((0.5,1,0.5))
            voxel_size = torch.tensor((voxel_size[0],voxel_size[2],voxel_size[1]))
            voxel_size = voxel_size/(voxel_size.sum())

            ## move vertices back to world space
            vertices = self.origin.view(1, 3).cpu().numpy() + vertices * voxel_size.view(1, 3).cpu().numpy()

            # f = lambda x,y,z: self.surface_mlp(torch.Tensor((x,y,z)))[0]

            # vertices, triangles = mcubes.marching_cubes_func((-2,-2,-2),(2,2,2),sample_density,sample_density,sample_density,f,0)
            ##faces = faces +1
            ##try:
                ##mcUtils.save_obj(vertices,normals,faces,output_dir=f"./",file_name=f"threshold_{x}.obj")
            mcubes.export_obj(vertices,triangles,f"threshold_{x}.obj")
            render_mesh(vertices,triangles,torch.tensor(0),f"threshold_{x}_render",profiler=profiler)
            ##except:
                #print(f"x is not able to thresholded the marching cubes")

        return vertices,faces,normals


    # def get_mesh(self) -> Mesh:
    #     """Extracts a mesh using marching cubes."""

    #     device = self.values.device

    #     # run marching cubes on CPU
    #     tsdf_values_np = self.values.clamp(-1, 1).cpu().numpy()
    #     print(f"tsdf value np: {tsdf_values_np}")
    #     vertices, faces, normals, _ = measure.marching_cubes(tsdf_values_np, level=0, allow_degenerate=False)

    #     vertices_indices = np.round(vertices).astype(int)
    #     colors = self.colors[vertices_indices[:, 0], vertices_indices[:, 1], vertices_indices[:, 2]]
    #     # move back to original device
    #     vertices = torch.from_numpy(vertices.copy()).to(device)
    #     faces = torch.from_numpy(faces.copy()).to(device)
    #     normals = torch.from_numpy(normals.copy()).to(device)

    #     # move vertices back to world space
    #     vertices = self.origin.view(1, 3) + vertices * self.voxel_size.view(1, 3)

    #     return Mesh(vertices=vertices, faces=faces, normals=normals, colors=colors)
    
    @classmethod
    def export_mesh(cls, mesh: Mesh, filename: str):
        """Exports the mesh to a file.
        We use pymeshlab to export the mesh as a PLY file.

        Args:
            mesh: The mesh to export.
            filename: The filename to export the mesh to.
        """
        vertex_matrix = mesh.vertices.cpu().numpy().astype("float64")
        face_matrix = mesh.faces.cpu().numpy().astype("int32")
        v_normals_matrix = mesh.normals.cpu().numpy().astype("float64")
        v_color_matrix = mesh.colors.cpu().numpy().astype("float64")
        # colors need an alpha channel
        v_color_matrix = np.concatenate([v_color_matrix, np.ones((v_color_matrix.shape[0], 1))], axis=-1)

        # create a new Mesh
        m = pymeshlab.Mesh(
            vertex_matrix=vertex_matrix,
            face_matrix=face_matrix,
            v_normals_matrix=v_normals_matrix,
            v_color_matrix=v_color_matrix,
        )
        # create a new MeshSet
        ms = pymeshlab.MeshSet()
        # add the mesh to the MeshSet
        ms.add_mesh(m, "mesh")
        # save the current mesh
        ms.save_current_mesh(filename)

   
    
    def integrate_tri_tsdf(
        self,
        divisions: int,
        depth_images: TensorType["batch", 3, "height", "width"],
        depth_images_outside: TensorType["batch", 3, "height", "width"],
        depth_images_inside: TensorType["batch", 3, "height", "width"],
        ray_origins: TensorType["batch", 3, "height", "width"],
        surface_normals: TensorType["batch", 3, "height", "width"],
        ray_direction: TensorType["batch",3,"height","width"],
        loss_weights: Tuple[float,float,float,float],
        profiler,
        ##color_images: Optional[TensorType["batch", 3, "height", "width"]] = None,
        mask_images: Optional[TensorType["batch", 1, "height", "width"]] = None,
    ):
        """Integrates a batch of depth images into the TSDF.

        Args:
            c2w: The camera extrinsics.
            K: The camera intrinsics.
            depth_images: The depth images to integrate.
            color_images: The color images to integrate.
            mask_images: The mask images to integrate.
        """
        if mask_images is not None:
            raise NotImplementedError("Mask images are not supported yet.")
        
        batches = torch.linspace(0,depth_images.shape[0],divisions,dtype=torch.int)

        self.surface_mlp.train(True)

        surface_loss_fn = torch.nn.L1Loss()

        print("###### NEW IMAGE ######")

        ##debug
        ##torch.set_printoptions(profile="full")
        device = depth_images.device

        print(f"shape comparison = outputs {depth_images.shape}. normalTruth {surface_normals.shape}")
        with torch.enable_grad():
            for n in range(1):
                surf_loss_sum = 0
                norm_reg_loss_avg = 0 
                sum_losses = 0
                norm_smooth_loss = 0
                x = 0
                # Sequentially update the TSDF...
                while x < batches.shape[0] - 1:
                    surface_points = depth_images[batches[x]:batches[x+1]]
                    outside_points = depth_images_outside[batches[x]:batches[x+1]]
                    inside_points = depth_images_inside[batches[x]:batches[x+1]]
                    normal_gt = surface_normals[batches[x]:batches[x+1]]
                    origins = ray_origins[batches[x]:batches[x+1]]
                    directions = ray_direction[batches[x]:batches[x+1]]

                    # print(f"Depth images: {depth_images[i]}")
                    # print(f"surface images: {surface_points[i]}")
                
                    ##number of groups the image is split into.
                    ##Indexing means this value cannot be lower than 2
                    # group = 25
                    counter = 0
                    # spaced_array = np.linspace(0,surface_points.shape[0],group,dtype=int)
                    # next_idx = 1
                    # for x in spaced_array:
                        #if(next_idx <group):
                    inputs = torch.cat((surface_points,
                                        outside_points,
                                        inside_points))
                    if torch.isnan(inputs).any():
                        print("ml inputs contain nan")

                    ##output dims: (surface [:,0], normal [:,1-3])

                    mlp_prediction_surface = self.surface_mlp(surface_points).to(device)
                    mlp_prediction_outside = self.surface_mlp(outside_points).to(device)
                    mlp_prediction_inside = self.surface_mlp(inside_points).to(device)
                    mlp_prediction_origins = self.surface_mlp(origins).to(device)


                    normal_truth = normal_gt

                    if torch.isnan(mlp_prediction_surface).any():
                        print("mloutputs contain nan")

                    #surface_loss_value = self.surface_loss(mlp_prediction_surface,mlp_prediction_outside,mlp_prediction_inside,mlp_prediction_origins)
                    surface_loss_value = (self.surface_surface_loss(mlp_prediction_surface) + self.inside_loss(mlp_prediction_inside)+ self.outside_loss(mlp_prediction_outside))
                    # input of surface normal part of prediction
                    normal_consistency_value = self.normal_consistency_loss(mlp_prediction_outside[:,1:], mlp_prediction_inside[:,1:], normal_reg_constant = 10)
                    smoothness_loss = self.normal_smoothness_loss(mlp_prediction_surface,normal_truth)
                    orientation_loss = self.normal_orientation_loss(mlp_prediction_surface[:,1:],directions)
                    
                    profiler.add_scalar("Loss/SurfaceLoss", surface_loss_value.sum()/(mlp_prediction_surface[:,0].shape[0]))
                    profiler.add_scalar("Loss/NormalRegularisation", normal_consistency_value.sum()/(mlp_prediction_surface[:,0].shape[0]))
                    profiler.add_scalar("Loss/NormalSmoothnessLoss", smoothness_loss.sum()/(mlp_prediction_surface[:,0].shape[0]))
                    profiler.add_scalar("Loss/NormalOrientationLoss", orientation_loss.sum()/(mlp_prediction_surface[:,0].shape[0]))

                    # surf_loss_sum += surface_loss_value
                    # norm_reg_loss_avg += normal_consistency_value
                    # norm_smooth_loss +=smoothness_loss


                    surface_loss_value *= loss_weights[0]
                    normal_consistency_value *= loss_weights[1]
                    smoothness_loss *= loss_weights[2]
                    orientation_loss *= loss_weights[3]

                    tot_loss = (surface_loss_value) + (normal_consistency_value) + (smoothness_loss) + orientation_loss
                    
                    profiler.add_scalar("LossContribution/SurfaceLoss", surface_loss_value.sum()/tot_loss.sum())
                    profiler.add_scalar("LossContribution/NormalRegularisation", normal_consistency_value.sum()/tot_loss.sum())
                    profiler.add_scalar("LossContribution/NormalSmoothnessLoss", smoothness_loss.sum()/tot_loss.sum())
                    profiler.add_scalar("LossContribution/NormalOrientationLoss", orientation_loss.sum()/tot_loss.sum())

                    
                    ##TEMPORARY. Allows nan failures to be displayed in tensorboard graphs
                    if torch.isnan(mlp_prediction_surface).any():
                        print("mloutputs contain nan")
                        quit()

                    #tot_loss *=1
                    counter+=1
                    #print(f"outputs = {outputs}")
                    sum_losses+=tot_loss.sum()


                    ##if(next_idx==1): print(F"Losses: {tot_loss}, {surface_loss_value}, {normal_consistency_value}, {smoothness_loss}")

                    profiler.add_scalar("Loss/SumLoss", tot_loss.sum()/(mlp_prediction_surface[:,0].shape[0]))
                    
                    ##Why not try this for a set
                    self.optimiser.zero_grad(set_to_none=True)
                    
                    tot_loss.sum().backward()

                    self.optimiser.step()
                    
                    del mlp_prediction_surface
                    del mlp_prediction_inside
                    del mlp_prediction_outside
                    del mlp_prediction_origins
                    del inputs

                    #increment batch group
                    x+=1
                
                    ##next_idx += 1
    
                    #input()
                            
                print(f"avgloss total ---> {sum_losses/depth_images[:,0].shape[0]}")
                print(f"avgloss surf ---> {surf_loss_sum/depth_images[:,0].shape[0]}")
                print(f"avgloss normreg-> {norm_reg_loss_avg/depth_images[:,0].shape[0]}")
                print(f"avgloss normSmoo-> {norm_smooth_loss/depth_images[:,0].shape[0]}")
                print(f"avgloss norm orient-> {norm_smooth_loss/depth_images[:,0].shape[0]}")
                print(f"ray samples per batch = {surface_points.shape[0]}")


    def gpu_usage_record(self):
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)

        profiler.add_scalar("gpu-usage",a/r)

    def normal_orientation_loss(self, predicted_normals: torch.Tensor, gt_direction: torch.Tensor):
        ## dot product can be done through torch.tensordot for multiple elements
        dot_prod = predicted_normals*gt_direction
        dot_prod = dot_prod.sum(dim=1)
        #dot_prod = torch.tensordot(predicted_normals,gt_direction,dims=2)
        loss = torch.max(torch.zeros_like(dot_prod),dot_prod)
        #return torch.sum(loss)
        return loss

    def normal_smoothness_loss(self,output_prediction: torch.Tensor, expected_outputs: torch.tensor):
        # output_divider = int(output_prediction.shape[0]/3)
        samples = output_prediction[:,1:]

        samples = safe_normalize(samples)
        expected_outputs = safe_normalize(expected_outputs)

        ##If normal is [0,0,0] replace the nan from normalising with a 0. This will give max loss against the, 
        # ideally, not zero norm from the NeRF.
        if torch.isnan(expected_outputs).any():
            print("expected outputs returned a NAN Value")
            expected_outputs = torch.nan_to_num(expected_outputs)

        regularity = samples-expected_outputs
        regularity = regularity**2
        loss = regularity.sum(dim=1)
        return loss


    ###uses finite difference to approximate a series of normals between the 16th and 84th percentile depths.
    ## This function presumes input of only the normals outputted from the mlp
    def normal_consistency_loss(self, output_pred_outside: torch.Tensor, output_pred_inside: torch.Tensor, normal_reg_constant: int = 10):
        linear_spaces = torch.linspace(0,1,normal_reg_constant).cuda()

        ## Normalise outputs from mlp 
        normal_outside = safe_normalize(output_pred_outside)
        if torch.isnan(normal_outside).any():
            sum_nan = torch.isnan(normal_outside).sum()
            print(f"Nan value calculated from mlp normal. Outside has {sum_nan}/{normal_outside.shape} nans ")
            normal_outside = torch.nan_to_num(normal_outside)
        
        normal_inside = safe_normalize(output_pred_inside)
        if torch.isnan(normal_inside).any():
            sum_nan = torch.isnan(normal_inside).sum()
            print(f"Nan value calculated from mlp normal. Inside has {sum_nan}/{normal_inside.shape} nans ")
            normal_inside = torch.nan_to_num(normal_inside)

        norm_difference = (normal_inside - normal_outside)

        norm_difference = norm_difference[None,:,:]
        norm_difference = norm_difference.expand(normal_reg_constant,-1,-1).cuda()
        if torch.isnan(norm_difference).any():
            print("difference between inside and outside norms has a nan! Training of the normals has failed.")

        outside_expanded = normal_outside[None,:,:]

        ##This can be removed. Normals sampled from tsdf.
        outside_expanded = outside_expanded.expand(1,-1,-1)
        linear_spaces_exp = torch.empty_like(norm_difference)
        counter =0
        for s in linear_spaces:
            linear_spaces_exp[counter,:,:] = s
            counter = counter+1
        linear_spaces_exp = (linear_spaces_exp*norm_difference)


        ## add all the samples per ray together. If all the same normal_regularity will = 10 so
        ## error will be 0.
        normal_samples = outside_expanded + linear_spaces_exp
        normal_samples = normal_samples.sum(dim=0)
        normal_samples = normal_samples.squeeze()

        normal_regularity = torch.linalg.norm(normal_samples,dim=1)
        normal_regularity = normal_regularity-normal_reg_constant
        ##normal_regularity = normal_regularity.sum(dtype=torch.float32)

        return normal_regularity**2
    
    def outside_loss(self,output_prediction_outside: torch.Tensor):
        surface_outside = output_prediction_outside[:,0]
        surface_loss_value = (surface_outside - (torch.ones_like(surface_outside)*0.1))**2
        return surface_loss_value
    
    def inside_loss(self,output_prediction_inside: torch.Tensor):
        surface_inside = output_prediction_inside[:,0]
        surface_loss_value = (surface_inside + (torch.ones_like(surface_inside)*0.1))**2
        return surface_loss_value
    def surface_surface_loss(self,output_prediction_surface: torch.Tensor):
        surface = output_prediction_surface[:,0]
        surface_loss_value = surface **2
        return surface_loss_value

    def surface_loss(self,output_prediction_surface: torch.Tensor,
                     output_prediction_outside: torch.Tensor,
                     output_prediction_inside: torch.Tensor,
                     output_prediction_origins: torch.Tensor):
        ##divide back into 16,50,84 densities
        ##output_divider = int(output_prediction.shape[0]/3)
        surface_mid = output_prediction_surface[:,0]
        surface_outside = output_prediction_outside[:,0]
        surface_inside = output_prediction_inside[:,0]
        origins = output_prediction_origins[:,0]
        
        ##print(f"###########################\nsurface = {outputs_surface.shape}, outside = {outputs_outside.shape}, inside = {outputs_inside.shape}")
        ##print(f"surface values: {output_prediction[:,0]}")

        ##surface loss Li in NerfMeshing                            
        ##As we know the desired outputs for each of the depth measurements, we can easily calculate euclidian 
        ## distances to each
        surface_loss_value = (((surface_outside-(torch.ones_like(surface_outside)*0.1))**2) + 
                            surface_mid**2 + 
                            ((surface_inside+(torch.ones_like(surface_outside)*0.1))**2))

        ##ray origins are always outside the surface value
        #ray_origin_loss = ((origins - (torch.ones_like(origins)*0.1))**2)
        
        surface_loss_value = torch.tensor(surface_loss_value,dtype=torch.float32)

        if not torch.isfinite(surface_loss_value).any():
            print("help")

        ##print(f"Individual surface Loss: {surface_loss_value}")

        ##more space needed to hold total loss value
        # surface_loss_value = surface_loss_value.sum(dtype=torch.float32)
        # surface_loss_value += ray_origin_loss.sum(dtype=torch.float32)

        return surface_loss_value
    

@torch.enable_grad()
def export_ssan(
    pipeline: Pipeline,
    output_dir: Path,
    downscale_factor: int = 2,
    depth_output_name: str = "depth",
    rgb_output_name: str = "rgb",
    resolution: Union[int, List[int]] = field(default_factory=lambda: [256, 256, 256]),
    batch_size: int = 2,
    use_bounding_box: bool = True,
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    loss_weights: Tuple[float,float,float,float] = (0.0001,0.00001,0.000001,0.00001),
    batch_splits: int = 30,
    epochs: int = 3,
    nerf_image_path: str = "",
    ray_limit: int = 77000000
):
    """Export a TSDF mesh from a pipeline.

    Args:
        pipeline: The pipeline to export the mesh from.
        output_dir: The directory to save the mesh to.
        downscale_factor: Downscale factor for the images.
        depth_output_name: Name of the depth output.
        rgb_output_name: Name of the RGB output.
        resolution: Resolution of the TSDF volume or [x, y, z] resolutions individually.
        batch_size: How many depth images to integrate per batch.
        use_bounding_box: Whether to use a bounding box for the TSDF volume.
        bounding_box_min: Minimum coordinates of the bounding box.
        bounding_box_max: Maximum coordinates of the bounding box.
        loss_weights: DEBUG: Change the weights of the losses applied during training of the tsdf.\n surface, Normal Consist., Normal smooth., Normal orient.
    """


    start_state = pipeline.state_dict()

    device = pipeline.device
    # X, Y, Z = np.mgrid[:30, :30, :30]
    # u = (X-15)**2 + (Y-15)**2 + (Z-15)**2 - 8**2
    # print(u.shape)
    # vertices, triangles = mcubes.marching_cubes(u, 0)
    # mcubes.export_obj(vertices, triangles, 'sphere.obj')


    dataparser_outputs = pipeline.datamanager.train_dataset._dataparser_outputs  # pylint: disable=protected-access

    # initialize the TSDF volume
    if not use_bounding_box:
        aabb = dataparser_outputs.scene_box.aabb
    else:
        aabb = torch.tensor([bounding_box_min, bounding_box_max])
    if isinstance(resolution, int):
        volume_dims = torch.tensor([resolution] * 3)
    elif isinstance(resolution, List):
        volume_dims = torch.tensor(resolution)
    else:
        raise ValueError("Resolution must be an int or a list.")
    tsdf_surface = TSDFfromSSAN.from_aabb(aabb, volume_dims=volume_dims)
    # move TSDF to device
    tsdf_surface.to(device)

    ##Tensorboard profiler 
    profiler = SummaryWriter(log_dir=f"{output_dir}\\{time.time()}")

    profiler.add_text(tag=f"Loss Weights",text_string= f"{loss_weights}")
    profiler.add_text(tag=f"Epochs",text_string= f"{epochs}")
    profiler.add_text(tag=f"Batch Splits",text_string= f"{batch_splits}")
    profiler.add_text(tag=f"Resolution",text_string= f"{resolution}")
    profiler.add_text(tag=f"Output Dir",text_string= f"{output_dir}")
    profiler.add_text(tag=f"Resolution",text_string= f"{resolution}")



    ##Use a new instance of the nerfacto model with 3 depth samplers in ray outputs.
    old_model = pipeline._model
    old_model_states = old_model.state_dict()

    pipeline._model = NerfactoModelTriDepth(
        config=old_model.config, scene_box=old_model.scene_box, num_train_data=old_model.num_train_data
    )
    ##This is vital. Allows the new model to read outputs correctly. Prevents the massively blurry input images problem
    pipeline.model.load_state_dict(old_model_states, True)

    ## Transplanting old varibles to new model. May be worth seeing what is necessary however will take some time. 
    pipeline.model.collider = old_model.collider
    pipeline.model._modules = old_model._modules
    pipeline.model._buffers = old_model._buffers
    pipeline.model._parameters = old_model._parameters
    pipeline.model.field = old_model.field

    del old_model
    del old_model_states

    pipeline.cuda()
    # camera per image supplied
    cameras = dataparser_outputs.cameras.to(device)

    depth_images_50 = torch.tensor([0])
    depth_images_16 = torch.tensor([0])
    depth_images_84 = torch.tensor([0])
    surface_normals = torch.tensor([0])
    ray_origins = torch.tensor([0])
    ray_directions = torch.tensor([0])
    ray_cam_inds = torch.tensor([0])

    if nerf_image_path == "":

        #we turn off distortion when populating the TSDF
        color_images, depth_images_50, depth_images_16, depth_images_84, surface_normals, ray_origins,ray_directions,ray_cam_inds = render_trajectory_tri_tsdf(
            pipeline,
            cameras,
            rgb_output_name=rgb_output_name,
            surface_depth_output_name=depth_output_name,
            outside_depth_output_name= "depth_16",
            inside_depth_output_name= "depth_84",
            rendered_resolution_scaling_factor=1.0 / downscale_factor,
            disable_distortion=True,
            # use_aabb=True,
            # bounding_box_min=bounding_box_min,
            # bounding_box_max=bounding_box_max
        )
        # Normal Sampling##

        # 10 in NeRF meshing paper. Can be altered though would require altering of hyperparameter in loss function as well
        # Nc in eq 12 of nerfmeshing.

        depth_images_50 = torch.tensor(np.array(depth_images_50), device=device)
        depth_images_16 = torch.tensor(np.array(depth_images_16), device=device)
        depth_images_84 = torch.tensor(np.array(depth_images_84), device=device)
        print("depth_images shape: {depth_images_50.shape}")
        surface_normals = torch.tensor(np.array(surface_normals), device=device)
        ray_origins = torch.tensor(np.array(ray_origins), device=device)
        ray_directions = torch.tensor(np.array(ray_directions), device=device)
        ray_cam_inds = torch.tensor(np.array(ray_cam_inds), device=device)
        color_images = torch.tensor(np.array(color_images), device=device)
        base_dir = os.path.curdir
        try:
            os.mkdir(f"./ssan/LastRender")
        except:
            print("directory Exists")
        
        os.chdir("./ssan/LastRender")

        ##with open("/test_arrays/depth_images_50","wb") as f:
        np.save("depth_images_50.npy",arr=np.array(depth_images_50.cpu()))
        np.save("depth_images_16.npy",arr=np.array(depth_images_16.cpu()))
        np.save("depth_images_84.npy",arr=np.array(depth_images_84.cpu()))
        np.save("surface_normals.npy",arr=np.array(surface_normals.cpu()))
        np.save("ray_origins.npy",arr=np.array(ray_origins.cpu()))
        np.save("ray_directions.npy",arr=np.array(ray_directions.cpu()))
        np.save("ray_cam_inds.npy",arr=np.array(ray_cam_inds.cpu()))
        np.save("color_images.npy",arr=np.array(color_images.cpu()))

        os.chdir(os.pardir)
        os.chdir(os.pardir)
        # print(f"Arrays saved to {os.curdir}")
    ## Use prerendered nerf image data for depth etc.
    else:
        base_dir = os.curdir
        os.chdir(nerf_image_path)
        depth_images_50 = torch.Tensor(np.load("depth_images_50.npy")).to(device)
        depth_images_16 = torch.Tensor(np.load("depth_images_16.npy")).to(device)
        depth_images_84 = torch.Tensor(np.load("depth_images_84.npy")).to(device)
        surface_normals = torch.Tensor(np.load("surface_normals.npy")).to(device)
        ray_origins = torch.Tensor(np.load("ray_origins.npy")).to(device)
        ray_directions = torch.Tensor(np.load("ray_directions.npy")).to(device)
        ray_cam_inds = torch.Tensor(np.load("ray_cam_inds.npy")).to(device)
        color_images = torch.Tensor(np.load("color_images.npy")).to("cpu")
        os.chdir(os.pardir)
        os.chdir(os.pardir)

    for x in np.linspace(0,60,5,dtype=int):
        profiler.add_image("Data/50 Depth Image/:",depth_images_50[x,:,:].cpu().numpy().T.swapaxes(1,2),global_step=x)
        profiler.add_image("Data/16 Depth Image/:",depth_images_16[x,:,:].cpu().numpy().T.swapaxes(1,2),global_step=x)
        profiler.add_image("Data/84 Depth Image/:",depth_images_84[x,:,:].cpu().numpy().T.swapaxes(1,2),global_step=x)
        profiler.add_image("Data/Normals Image/:",surface_normals[x,:,:].cpu().numpy().T.swapaxes(1,2),global_step=x)
        profiler.add_image("Data/Ray Directions/: ",ray_directions[x,:,:].cpu().numpy().T.swapaxes(1,2),global_step=x)
        profiler.add_image("Data/Colour Image/: ",color_images[x,:,:].cpu().numpy().T.swapaxes(1,2),global_step=x)
        
    dataset = SSANDataset(depth_images_50, depth_images_16, depth_images_84, surface_normals, ray_origins,ray_directions,ray_cam_inds)
    
    del depth_images_50
    del depth_images_16
    del depth_images_84
    del surface_normals
    del ray_origins
    del ray_directions
    del ray_cam_inds
    
    dataset.depth_to_point()
    for x in np.linspace(0,60,5,dtype=int):
        profiler.add_image("Data/50 Depth Point/:",dataset.depth_50[x,:,:].cpu().numpy().T.swapaxes(1,2),global_step=x)
        profiler.add_image("Data/16 Depth Point/:",dataset.depth_16[x,:,:].cpu().numpy().T.swapaxes(1,2),global_step=x)
        profiler.add_image("Data/84 Depth Point/:",dataset.depth_84[x,:,:].cpu().numpy().T.swapaxes(1,2),global_step=x)
        


    dataset.to_2d_array()
    
    ## Remove rays which terminate outside the bounding box
    bounding_box_min = torch.tensor(bounding_box_min).to(device)
    bounding_box_max = torch.tensor(bounding_box_max).to(device)

    remove_rays_outside_AABB(dataset,bounding_box_min,bounding_box_max)
    dataset.to_aabb_bounding_box(bounding_box_min,bounding_box_max)

    print(f"Current dir: {os.listdir(os.curdir)}")
    



    print(f"{dataset.depth_50.shape}")




    remove_inf_and_nan(dataset)
    dataset.trim_data(ray_limit)

    ##adjust mesh construction coords
    print(tsdf_surface.voxel_coords.shape)
    tsdf_surface.voxel_coords = (tsdf_surface.voxel_coords.T -bounding_box_min).T
    tsdf_surface.voxel_coords = (tsdf_surface.voxel_coords.T/(bounding_box_max-bounding_box_min)).T
    print(tsdf_surface.voxel_coords.shape)


    ##split dataset into two equal randomised halfs, then carry forward one.
    # split_data = torch_data.RandomSampler(dataset,False,int(dataset.depth_50.shape[0]/2))
    print(dataset.depth_50.shape)


    color_images = torch.tensor(np.array(color_images.cpu()), device=device).permute(0, 3, 1, 2)  # shape (N, 3, H, W)

    CONSOLE.print("Integrating the Surface TSDF")

    ##normal_samples = torch.reshape(normal_samples,[270*270*480,3])



    ##batch_size number of images worth of rays randomly selected.
    ## batch_size * img_width * img_height 
    num_rays = int(batch_size) * int(cameras.cx.mean()) * int(cameras.cy.mean())
    divisions = batch_splits

    profiler.add_text(tag=f"MLP Parameters",text_string= f"{tsdf_surface.surface_mlp.__str__()}")

   ##Batches changed from original tsdf integration to accomodate 2d input
    for e in range(epochs):
        # shuffle rays randomly
        dataset.shuffle_data()
        print(f"Dataset shuffled")
        print(f"### EPOCH {e}####\n################")
        for i in range(0, dataset.depth_50.shape[0], num_rays):
            tsdf_surface.integrate_tri_tsdf(
                divisions,
                dataset.depth_50[i : i + num_rays],
                dataset.depth_16[i : i + num_rays],
                dataset.depth_84[i : i + num_rays],
                dataset.ray_origins[i : i + num_rays],
                dataset.surface_normals[i:i+num_rays],
                dataset.ray_directions[i:i + num_rays],
                loss_weights,
                profiler,
                ##color_images=color_images[i : i + batch_size],
            )

    surfaceHyperparameter = 0.1

    CONSOLE.print("Computing Mesh")


    verts,faces,norms = tsdf_surface.get_mesh(output_dir,profiler=profiler)

    ##profiler.add_mesh("Mesh: ",vertices=torch.tensor(verts).unsqueeze(0),faces=torch.tensor(faces).unsqueeze(0))

    print(f"surface Values: {tsdf_surface.values.shape}")
    profiler.close()
    return 0

    tsdf_surface.export_mesh(mesh_surface, filename=str(output_dir / "ssan_mesh_surface.ply"))

    CONSOLE.print("Saved SSAN Mesh")

##Works only over 2 Dimensional data##
def remove_rays_outside_AABB(data: SSANDataset, AABB_min: torch.Tensor, AABB_max: torch.Tensor):
    greater_than_min = torch.where(data.depth_50 >= AABB_min,True,False)
    less_than_max = torch.where(data.depth_50 <= AABB_max,True,False)


    valid_datums = torch.any((torch.bitwise_and(greater_than_min,less_than_max)),dim=1)

    _d50 = data.depth_50[valid_datums]
    data.depth_50 = _d50
    del _d50

    _d16 = data.depth_16[valid_datums]
    data.depth_16 = _d16
    del _d16

    _d84 = data.depth_84[valid_datums]
    data.depth_84 = _d84
    del _d84

    _norms = data.surface_normals[valid_datums]
    data.surface_normals = _norms
    del _norms

    _orig = data.ray_origins[valid_datums]
    data.ray_origins = _orig
    del _orig

    _dir = data.ray_directions[valid_datums]
    data.ray_directions = _dir
    del _dir

    _cam = data.ray_cam_inds[valid_datums]
    data.ray_cam_inds = _cam
    del _cam

##Works only over 2 Dimensional data##
def remove_inf_and_nan(data: SSANDataset):
    is_finite = torch.isfinite(data.depth_50)
    is_inside = torch.where(data.depth_50.abs() <= 1, True, False)

    ### how many 3d vectors are there?
    finite_length = is_finite.sum()/3
    valid_datums = torch.any((torch.bitwise_and(is_finite,is_inside)),dim=1)
    
    print(f"Rays used {data.depth_16.shape}/{data.depth_50[valid_datums].shape}")
    _d50 = data.depth_50[valid_datums]
    data.depth_50 = _d50
    del _d50

    _d16 = data.depth_16[valid_datums]
    data.depth_16 = _d16
    del _d16

    _d84 = data.depth_84[valid_datums]
    data.depth_84 = _d84
    del _d84

    _norms = data.surface_normals[valid_datums]
    data.surface_normals = _norms
    del _norms

    _orig = data.ray_origins[valid_datums]
    data.ray_origins = _orig
    del _orig

    _dir = data.ray_directions[valid_datums]
    data.ray_directions = _dir
    del _dir

    _cam = data.ray_cam_inds[valid_datums]
    data.ray_cam_inds = _cam
    del _cam

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
            raise ValueError("depth_50 not the right shape. Try using depth_to_point before.")

        self.depth_50 -= bounding_box_min
        self.depth_50 /= (bounding_box_max-bounding_box_min)

        self.depth_16 -= bounding_box_min
        self.depth_16 /= (bounding_box_max-bounding_box_min)

        self.depth_84 -= bounding_box_min
        self.depth_84 /= (bounding_box_max-bounding_box_min)


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




    def __len__(self):
        return len(self.depth_50)

