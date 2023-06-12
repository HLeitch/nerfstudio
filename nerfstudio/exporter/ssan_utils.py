"""
SSAN utils.
"""

# pylint: disable=no-member

from __future__ import annotations

import time as time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import GPUtil
import numpy as np
import pymeshlab
import tinycudann as tcnn
import torch
import torch.nn.functional as F
from rich.console import Console
from skimage import measure
from torch.utils.tensorboard import SummaryWriter
from torchtyping import TensorType

import nerfstudio.fields.nerfacto_field
from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.exporter.exporter_utils import (
    Mesh,
    render_trajectory,
    render_trajectory_tri_tsdf,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.nerfacto import NerfactoModelTriDepth
from nerfstudio.pipelines.base_pipeline import Pipeline

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
    surface_mlp: tcnn.NetworkWithInputEncoding
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
        voxel_coords = origin.view(3, 1, 1, 1) + grid * voxel_size.view(3, 1, 1, 1)

        # initialize the values and weights
        values = -torch.ones(volume_dims.tolist())
        normal_values = torch.zeros(volume_dims.tolist()+[3])
        normal_weights = torch.zeros(volume_dims.tolist()+[3])
        weights = torch.zeros(volume_dims.tolist())
        colors = torch.zeros(volume_dims.tolist() + [3])

        ##growth_factor = np.exp((np.log(2048)-np.log(16)/(15-1)))
        ###
        ##taken from nerfacto field parameters##
        surface_mlp = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=1 + 3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.382,

            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2 - 1,
                "seed": 210799
            },
        )
        optimiser = torch.optim.Adam(surface_mlp.parameters(), lr=0.005, betas=(0.9,0.99),eps=10e-15)

        # TODO: move to device

        return TSDFfromSSAN(voxel_coords, values, weights,normal_values,normal_weights, surface_mlp,optimiser, colors, voxel_size, origin)

    def get_mesh(self) -> Mesh:
        """Extracts a mesh using marching cubes."""

        device = self.values.device

        # run marching cubes on CPU
        tsdf_values_np = self.values.clamp(-1, 1).cpu().numpy()
        print(f"tsdf value np: {tsdf_values_np}")
        vertices, faces, normals, _ = measure.marching_cubes(tsdf_values_np, level=0, allow_degenerate=False)

        vertices_indices = np.round(vertices).astype(int)
        colors = self.colors[vertices_indices[:, 0], vertices_indices[:, 1], vertices_indices[:, 2]]
        # move back to original device
        vertices = torch.from_numpy(vertices.copy()).to(device)
        faces = torch.from_numpy(faces.copy()).to(device)
        normals = torch.from_numpy(normals.copy()).to(device)

        # move vertices back to world space
        vertices = self.origin.view(1, 3) + vertices * self.voxel_size.view(1, 3)

        return Mesh(vertices=vertices, faces=faces, normals=normals, colors=colors)

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
        c2w: TensorType["batch", 4, 4],
        K: TensorType["batch", 3, 3],
        depth_images: TensorType["batch", 1, "height", "width"],
        depth_images_outside: TensorType["batch", 1, "height", "width"],
        depth_images_inside: TensorType["batch", 1, "height", "width"],
        ray_origins: TensorType["batch", 3, "height", "width"],
        surface_normals: TensorType["batch", 3, "height", "width"],
        normal_samples: TensorType["batch", 3, "height", "width"],
        normal_regularity: TensorType["batch", 1, "height", "width"],
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

        batch_size = c2w.shape[0]
        shape = self.voxel_coords.shape[1:]

        # Project voxel_coords into image space...

        image_size = torch.tensor(
            [depth_images.shape[-1], depth_images.shape[-2]], device=self.device
        )  # [width, height]

        self.surface_mlp.train(True)

        loss = torch.nn.L1Loss()
        print("###### NEW IMAGE ######")

        ##debug
        ##torch.set_printoptions(profile="full")
        device = depth_images.device

        with torch.enable_grad():
            for n in range(1):
                surf_loss_sum = 0
                norm_reg_loss_avg = 0 
                sum_losses = 0
                # Sequentially update the TSDF...
                for i in range(batch_size):
                    
                    surface_points = depth_images[i]
                    surface_points = surface_points.reshape(3,-1).t()

                    outside_points = depth_images_outside[i]
                    outside_points = outside_points.reshape(3,-1).t()
                    inside_points = depth_images_inside[i]
                    inside_points = inside_points.reshape(3,-1).t()
                    
                    ##number of groups the image is split into.
                    ##Indexing means this value cannot be lower than 2
                    batches = 7

                    spaced_array = np.linspace(0,surface_points.shape[0],batches,dtype=int)
                    next_idx = 1
                    for x in spaced_array:
                        if(next_idx <batches):
                            inputs = torch.cat((surface_points[x:spaced_array[next_idx]],
                                                outside_points[x:spaced_array[next_idx]],
                                                inside_points[x:spaced_array[next_idx]]))
                            
                            ##output dims: (surface [:,0], normal [:,1-3])
                            outputs = self.surface_mlp(inputs).to(device)

                            if torch.isnan(outputs).any():
                                continue

                            surface_loss_value = self.surface_loss(outputs)
                            normal_smoothness_value = self.normal_smoothness_loss(outputs)

                            loss = (100*surface_loss_value) + (0.001*normal_smoothness_value)
                            #print(f"outputs = {outputs}")
                            sum_losses+=loss
                            surf_loss_sum += surface_loss_value
                            norm_reg_loss_avg += normal_smoothness_value


                            self.optimiser.zero_grad()
                            surface_loss_value.backward()
                            
                            next_idx += 1
            
                            #input()
                            
                print(f"avgloss image ---> {sum_losses/outputs[:,0].shape[0]}")
                print(f"avgloss surf ---> {surf_loss_sum/outputs[:,0].shape[0]}")
                print(f"avgloss normSmoo-> {norm_reg_loss_avg/outputs[:,0].shape[0]}")
                profiler.add_scalar("Loss/SumLoss", sum_losses/outputs[:,0].shape[0])
                profiler.add_scalar("Loss/SurfaceLoss", surf_loss_sum/outputs[:,0].shape[0])
                profiler.add_scalar("Loss/NormalRegularisation", norm_reg_loss_avg/outputs[:,0].shape[0])

            self.optimiser.step()


    ##uses finite difference to approximate a series of normals between the 16th and 84th percentile depths.
    def normal_smoothness_loss(self,output_prediction: torch.Tensor,normal_reg_constant: int = 10):
        linear_spaces = torch.linspace(0,1,10).cuda()
        output_divider = int(output_prediction.shape[0]/3)
                                
        normal_outside = output_prediction[output_divider:output_divider*2,1:]
        normal_outside = F.normalize(normal_outside,dim=1)
        ##print(normal_outside)
        normal_inside = output_prediction[output_divider*2:,1:]
        normal_inside = F.normalize(normal_inside,dim=1)


        norm_difference = (normal_inside - normal_outside)
        ##print(f"norm_difference: {norm_difference} \n{norm_difference.shape}")


        norm_difference = norm_difference[None,:,:]
        norm_difference = norm_difference.expand(10,-1,-1).cuda()

        outside_expanded = normal_outside[None,:,:]

        ##This can be removed. Normals sampled from tsdf.
        outside_expanded = outside_expanded.expand(1,-1,-1)
        linear_spaces_exp = torch.empty_like(norm_difference)
        counter =0
        for s in linear_spaces:
            linear_spaces_exp[counter,:,:] = s
            counter = counter+1
        linear_spaces_exp = (linear_spaces_exp*norm_difference)
        normal_samples = outside_expanded + linear_spaces_exp

        normal_samples = normal_samples.sum(dim=0)
        normal_samples = normal_samples.squeeze()
        normal_regularity = torch.linalg.norm(normal_samples,dim=1)
        normal_regularity = normal_regularity-normal_reg_constant
        normal_regularity = normal_regularity.sum(dtype=torch.float32)
        return normal_regularity


    def surface_loss(self,output_prediction: torch.Tensor): 
        ##divide back into 16,50,84 densities
        output_divider = int(output_prediction.shape[0]/3)
                                
        surface_mid = output_prediction[0:output_divider,0]
        surface_outside = output_prediction[output_divider:output_divider*2,0]
        surface_inside = output_prediction[output_divider*2:,0]
        ##print(f"###########################\nsurface = {outputs_surface.shape}, outside = {outputs_outside.shape}, inside = {outputs_inside.shape}")
        ##print(f"surface values: {output_prediction[:,0]}")

        ##surface loss Li in NerfMeshing                            
        ##As we know the desired outputs for each of the depth measurements, we can easily calculate euclidian 
        ## distances to each
        surface_loss_value = (((surface_outside-(torch.ones_like(surface_outside)/10))**2) + 
                            surface_mid**2 + 
                            ((surface_inside+(torch.ones_like(surface_outside)/10))**2))
        if torch.isnan(surface_loss_value).any():
            print("help")

        ##print(f"Individual surface Loss: {surface_loss_value}")

        ##more space needed to hold total loss value
        surface_loss_value = surface_loss_value.sum(dtype=torch.float32)

        return surface_loss_value
def export_ssan(
    pipeline: Pipeline,
    output_dir: Path,
    downscale_factor: int = 2,
    depth_output_name: str = "depth",
    rgb_output_name: str = "rgb",
    resolution: Union[int, List[int]] = field(default_factory=lambda: [256, 256, 256]),
    batch_size: int = 10,
    use_bounding_box: bool = True,
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
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
    """


    start_state = pipeline.state_dict()

    device = pipeline.device
    

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

    pipeline.cuda()
    # camera per image supplied
    cameras = dataparser_outputs.cameras.to(device)
    #print(f"Cameras 1: {cameras.camera_to_worlds} ")
    # we turn off distortion when populating the TSDF
    color_images, depth_images_50, depth_images_16, depth_images_84, surface_normals, ray_origins,ray_directions,ray_cam_inds = render_trajectory_tri_tsdf(
        pipeline,
        cameras,
        rgb_output_name=rgb_output_name,
        surface_depth_output_name=depth_output_name,
        outside_depth_output_name= "depth_16",
        inside_depth_output_name= "depth_84",
        rendered_resolution_scaling_factor=1.0 / downscale_factor,
        disable_distortion=True,
    )
    ## Normal Sampling##

    ## 10 in NeRF meshing paper. Can be altered though would require altering of hyperparameter in loss function as well
    ## Nc in eq 12 of nerfmeshing.
    linear_spaces = torch.linspace(0,1,1).cuda()
    depth_images_50 = torch.tensor(np.array(depth_images_50), device=device)
    depth_images_16 = torch.tensor(np.array(depth_images_16), device=device) 
    depth_images_84 = torch.tensor(np.array(depth_images_84), device=device)  
    print(depth_images_50.shape)
    surface_normals = torch.tensor(np.array(surface_normals), device=device)
    ray_origins = torch.tensor(np.array(ray_origins), device=device)
    ray_directions = torch.tensor(np.array(ray_directions), device=device)
    ray_cam_inds = torch.tensor(np.array(ray_cam_inds), device=device)

    print(f"ray: {ray_origins.shape}\ndepth: {depth_images_16.shape}")

    depth_images_50 = ray_origins+(ray_directions*depth_images_50)
    depth_images_16 = ray_origins+(ray_directions*depth_images_16)
    depth_images_84 = ray_origins+(ray_directions*depth_images_84)
    outside_positions = ray_origins+(ray_directions*depth_images_16)
    inside_positions = ray_origins+(ray_directions*depth_images_84)


    # print(f"outside: {outside_positions[:,0,0,0]}")
    # print(f"inside: {inside_positions[:,0,0,0]}")

    pos_difference = (inside_positions - outside_positions)
    # print(f"Pos_difference: {pos_difference[:,0,0,0]}")


    pos_difference = pos_difference[None,:,:,:,:]
    pos_difference = pos_difference.expand(10,-1,-1,-1,-1).cuda()

    outside_expanded = outside_positions[None,:,:,:,:]

    ##This can be removed. Normals sampled from tsdf.
    outside_expanded = outside_expanded.expand(1,-1,-1,-1,-1)
    linear_spaces_exp = torch.empty_like(pos_difference)
    counter =0
    for s in linear_spaces:
        linear_spaces_exp[counter,:,:,:,:] = s
        counter = counter+1
    # print(distance_expanded.shape)
    linear_spaces_exp = (linear_spaces_exp*pos_difference)
    normal_position_samples = outside_expanded + linear_spaces_exp
    
    #print(f"Samples: {distance_expanded[0,:,0,0,0]}\n, {distance_expanded[0,:,0,0,1]}\n,{distance_expanded[0,:,0,0,2]}")

    # print(f"Normal position Samples: {normal_position_samples[0,:,0,0,0]}\n, {normal_position_samples[0,:,0,0,1]}\n,{normal_position_samples[0,:,0,0,2]}")
    print(f"normal positions shape: {normal_position_samples.shape}")
    
    ##Slicing to fit onto gpu
    normal_samples = torch.zeros_like(normal_position_samples)


    i = 0

    # # memory usage seems stable
    # for n in torch.chunk(normal_position_samples,normal_position_samples.shape[0],dim=0):
    #     n = n.squeeze()
    #     print(f"normals at the {i} position in ray being calculated")
    #     j=0
    #     for m in torch.chunk(n,n.shape[0],dim=0):
    #         m.squeeze
    #         ##pipeline.model.field.density_fn(n)
    #         ray_samples = RaySamples(
    #             frustums=Frustums(
    #                 origins=m,
    #                 directions=torch.ones_like(m),
    #                 starts=torch.zeros_like(m[..., :1]),
    #                 ends=torch.zeros_like(m[..., :1]),
    #                 pixel_area=torch.ones_like(m[..., :1]),
    #             ),
    #             camera_indices= ray_cam_inds[i,j]
    #         )

    #         normal_slice = pipeline.model.field.forward(ray_samples,True)[FieldHeadNames.NORMALS]

    #         normal_samples[i,j,:,:,:] = normal_slice
    #         j = j+1

            
    #         del(normal_slice,ray_samples)

    #     i = i+1
    
    del(ray_cam_inds)

    print(f"normalsamples : {normal_samples.shape}")

    normal_samples = normal_samples.sum(dim=0)
    normal_samples.squeeze()

    ### Max regularity of 10 (as in paper) or the number of normal points sampled per ray. 
    normal_regularity = torch.linalg.norm(normal_samples,dim=3)[:,:,:,None]

    # print(normal_regularity)
    # print(normal_regularity.shape)
    # print(f"{normal_samples[22,12,5,:]} has magnitude {normal_regularity[22,12,5,:]}")
    # camera extrinsics and intrinsics
    c2w: TensorType["N", 3, 4] = cameras.camera_to_worlds.to(device)
    # make c2w homogeneous
    c2w = torch.cat([c2w, torch.zeros(c2w.shape[0], 1, 4, device=device)], dim=1)
    c2w[:, 3, 3] = 1
    K: TensorType["N", 3, 3] = cameras.get_intrinsics_matrices().to(device)
    color_images = torch.tensor(np.array(color_images), device=device).permute(0, 3, 1, 2)  # shape (N, 3, H, W)
    depth_images_50 = torch.tensor(depth_images_50, device=device).permute(0, 3, 1, 2)  # shape (N, 1, H, W)
    depth_images_16 = torch.tensor(depth_images_16, device=device).permute(0, 3, 1, 2)  # shape (N, 1, H, W)
    depth_images_84 = torch.tensor(depth_images_84, device=device).permute(0, 3, 1, 2)  # shape (N, 1, H, W)
    surface_normals = torch.tensor(surface_normals, device=device).permute(0,3,1,2) # shape (N, 1, H, W)
    normal_samples = torch.tensor(normal_samples, device=device).permute(0,3,1,2) # shape (N, 1, H, W)
    normal_regularity = torch.tensor(normal_regularity, device=device).permute(0,3,1,2)   # shape (N, 1, H, W)

    print(depth_images_50.shape)
    
    ray_origins = torch.tensor(ray_origins, device=device).permute(0, 3, 1, 2)  # shape (N, 1, H, W)
    # ray_directions = torch.tensor(np.array(ray_directions), device=device).permute(0, 3, 1, 2)  # shape (N, 1, H, W)
    # ray_cam_inds = torch.tensor(np.array(ray_cam_inds), device=device).permute(0, 3, 1, 2)  # shape (N, 1, H, W)
    CONSOLE.print("Integrating the Surface TSDF")


    ##profiler 
    profiler = SummaryWriter()
   
    for e in range(10):
        print(f"### EPOCH {e}####\n################")
        for i in range(0, len(c2w), batch_size):
            tsdf_surface.integrate_tri_tsdf(
                c2w[i : i + batch_size],
                K[i : i + batch_size],
                depth_images_50[i : i + batch_size],
                depth_images_16[i : i + batch_size],
                depth_images_84[i : i + batch_size],
                ray_origins[i : i + batch_size],
                surface_normals[i:i+batch_size],
                normal_samples[i:i+batch_size],
                normal_regularity[i:i+batch_size],
                profiler,
                ##color_images=color_images[i : i + batch_size],
            )

    surfaceHyperparameter = 0.1

    CONSOLE.print("Computing Mesh")


    mesh_surface = tsdf_surface.get_mesh()

    print(f"surface Values: {tsdf_surface.values.shape}")

    tsdf_surface.export_mesh(mesh_surface, filename=str(output_dir / "ssan_mesh_surface.ply"))

    CONSOLE.print("Saved SSAN Mesh")