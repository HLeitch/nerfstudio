"""
Script for exporting NeRF into other formats.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import open3d.visualization as o3dvis
import skimage as skimage
import torch
import tyro
from rich.console import Console
from torch.utils.tensorboard import SummaryWriter
from typing_extensions import Annotated, Literal

import nerfstudio.cameras.cameras as nscam
import nerfstudio.exporter.marching_cubes_utils as mcUtils
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.exporter import ssan_utils, texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import (collect_camera_poses,
                                                density_sampler,
                                                generate_point_cloud,
                                                get_mesh_from_filename)
from nerfstudio.exporter.object_renderer import render_mesh_to_tb
from nerfstudio.exporter.unit_tests import display_histogram_of_densities
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils import math as math
from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)


import json
import os
import sys
import typing
from collections import OrderedDict
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import open3d as o3d
import torch
import tyro
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.parallel_datamanager import \
    ParallelDataManager
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import (collect_camera_poses,
                                                generate_point_cloud,
                                                get_mesh_from_filename)
from nerfstudio.exporter.marching_cubes import \
    generate_mesh_with_multires_marching_cubes
from nerfstudio.fields.sdf_field import SDFField  # noqa
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""


def validate_pipeline(normal_method: str, normal_output_name: str, pipeline: Pipeline) -> None:
    """Check that the pipeline is valid for this exporter.

    Args:
        normal_method: Method to estimate normals with. Either "open3d" or "model_output".
        normal_output_name: Name of the normal output.
        pipeline: Pipeline to evaluate with.
    """
    if normal_method == "model_output":
        CONSOLE.print("Checking that the pipeline has a normal output.")
        origins = torch.zeros((1, 3), device=pipeline.device)
        directions = torch.ones_like(origins)
        pixel_area = torch.ones_like(origins[..., :1])
        camera_indices = torch.zeros_like(origins[..., :1])
        metadata = {"directions_norm": torch.linalg.vector_norm(directions, dim=-1, keepdim=True)}
        ray_bundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
            metadata=metadata,
        )
        outputs = pipeline.model(ray_bundle)
        if normal_output_name not in outputs:
            CONSOLE.print(f"[bold yellow]Warning: Normal output '{normal_output_name}' not found in pipeline outputs.")
            CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
            CONSOLE.print(
                "[bold yellow]Warning: Please train a model with normals "
                "(e.g., nerfacto with predicted normals turned on)."
            )
            CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
            CONSOLE.print("[bold yellow]Exiting early.")
            sys.exit(1)


@dataclass
class ExportPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    save_world_frame: bool = False
    """If set, saves the point cloud in the same frame as the original dataset. Otherwise, uses the
    scaled and reoriented coordinate space expected by the NeRF models."""

    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(
            pipeline.datamanager,
            (VanillaDataManager, ParallelDataManager),
        )
        if isinstance(pipeline.datamanager, VanillaDataManager):
            assert pipeline.datamanager.train_pixel_sampler is not None
            pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        crop_obb = None
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        if self.save_world_frame:
            # apply the inverse dataparser transform to the point cloud
            points = np.asarray(pcd.points)
            poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
            poses[:, :3, 3] = points
            poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
                torch.from_numpy(poses)
            )
            points = poses[:, :3, 3].numpy()
            pcd.points = o3d.utility.Vector3dVector(points)

        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
        o3d.t.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), tpcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")



@dataclass
class ExportTSDFMesh(Exporter):
    """
    Export a mesh using TSDF processing.
    """

    downscale_factor: int = 2
    """Downscale the images starting from the resolution used for training."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    resolution: Union[int, List[int]] = field(default_factory=lambda: [128, 128, 128])
    """Resolution of the TSDF volume or [x, y, z] resolutions individually."""
    batch_size: int = 10
    """How many depth images to integrate per batch."""
    use_bounding_box: bool = True
    """Whether to use a bounding box for the TSDF volume."""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    texture_method: Literal["tsdf", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'tsdf' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""
    refine_mesh_using_initial_aabb_estimate: bool = False
    """Refine the mesh using the initial AABB estimate."""
    refinement_epsilon: float = 1e-2
    """Refinement epsilon for the mesh. This is the distance in meters that the refined AABB/OBB will be expanded by
    in each direction."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        tsdf_utils.export_tsdf_mesh(
            pipeline,
            self.output_dir,
            self.downscale_factor,
            self.depth_output_name,
            self.rgb_output_name,
            self.resolution,
            self.batch_size,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            refine_mesh_using_initial_aabb_estimate=self.refine_mesh_using_initial_aabb_estimate,
            refinement_epsilon=self.refinement_epsilon,
        )

        # possibly
        # texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the tsdf export
            mesh = get_mesh_from_filename(
                str(self.output_dir / "tsdf_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )

@dataclass
class ExportPoissonMesh(Exporter):
    """
    Export a mesh using poisson surface reconstruction.
    """

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    save_point_cloud: bool = False
    """Whether to save the point cloud."""
    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    texture_method: Literal["point_cloud", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'point_cloud' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(
            pipeline.datamanager,
            (VanillaDataManager, ParallelDataManager),
        )
        if isinstance(pipeline.datamanager, VanillaDataManager):
            assert pipeline.datamanager.train_pixel_sampler is not None
            pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        else:
            crop_obb = None

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        torch.cuda.empty_cache()
        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")

        if self.save_point_cloud:
            CONSOLE.print("Saving Point Cloud...")
            o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
            print("\033[A\033[A")
            CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")

        CONSOLE.print("Computing Mesh... this may take a while.")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

        CONSOLE.print("Saving Mesh...")
        o3d.io.write_triangle_mesh(str(self.output_dir / "poisson_mesh.ply"), mesh)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Mesh")

        # This will texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the poisson reconstruction
            mesh = get_mesh_from_filename(
                str(self.output_dir / "poisson_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportMarchingCubesMesh(Exporter):
    """
    Export a mesh using marching cubes.
    EXAMPLE: ns-export marching-cubes --load-config [config path] --output-dir exports/mc/ --use-bounding-box True --bounding-box-min -0.25 -0.25 -0.25 --bounding-box-max 0.25 0.25 0.25 --num-samples=100 --save_mesh True --output-file-name example.obj
    """

    CONSOLE.print("Marching Cubes STARTED", highlight=True)

    num_samples: int = 100
    """Number of points to sample per axis. May result in less if outlier removal is used."""
    mc_level: int = int(10)
    """Threshold value for surfaces. Affects smoothness and amount of floaters. Higher = fewer floaters, more craters in object"""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    save_mesh: bool = True
    """Whether to save the point cloud."""
    output_file_name: str = "marching-cubes.obj"
    """Name of file output is saved to"""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    texture_method: Literal["point_cloud", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'point_cloud' or 'nerf'."""

    def validate_pipeline(self, pipeline: Pipeline) -> None:
        """Check that the pipeline is valid for this exporter."""

    @torch.no_grad()
    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _ = eval_setup(self.load_config)

        self.validate_pipeline(pipeline)

        # Increase the batchsize to speed up the evaluation.
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        densities = density_sampler(
            pipeline=pipeline,
            num_samples=self.num_samples,
            remove_outliers=self.remove_outliers,
            depth_output_name=self.depth_output_name,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
        )
        torch.cuda.empty_cache()

        verts, faces, normals, values = skimage.measure.marching_cubes(
            densities, level=self.mc_level, allow_degenerate=False
        )

        colours = np.zeros_like(verts)

        CONSOLE.print(f"[bold green]:white_check_mark: Generated Marching Cube representation!!")

        if self.save_mesh:
            ##Other programs for model veiwing read from 1. Python indexes from 0
            facesReindex = faces + 1

            mcUtils.save_obj(verts, normals, facesReindex, self.output_dir, self.output_file_name)


@dataclass
class ExportSamuraiMarchingCubes(Exporter):
    """
    Export a mesh using the extraction technique described in SAMURAI (https://markboss.me/publication/2022-samurai/)
    Largely adapted from the repo created of project.
    """

    CONSOLE.print("Samurai Marching Cubes STARTED", highlight=True)

    num_samples_mc: int = 100
    """Number of points to sample per axis. May result in less if outlier removal is used."""
    num_samples_points: int = 2000000
    """Number of points sampled on naive mesh"""
    mc_level: float = float(10)
    """Threshold value for surfaces. Affects smoothness and amount of floaters. Higher = fewer floaters, more craters in object"""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    ray_depth_length: float = float(0.5)
    """Maximum distance sampled to surface."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    save_mesh: bool = True
    """Whether to save the point cloud."""
    output_file_name: str = "marching-cubes.obj"
    """Name of file output is saved to"""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    texture_method: Literal["point_cloud", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'point_cloud' or 'nerf'."""

    def validate_pipeline(self, pipeline: Pipeline) -> None:
        """Check that the pipeline is valid for this exporter."""

    @torch.no_grad()
    def main(self) -> None:
        """Export mesh"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _, pipeline, _,_ = eval_setup(self.load_config)

        self.validate_pipeline(pipeline)

        tb_file = SummaryWriter(self.output_dir.__str__()+f"/{time.time()}")

        ##record parameters to tensorboard record
        tb_file.add_text(f"num_samples_mc",f"{self.num_samples_mc}")
        tb_file.add_text(f"num_samples_points",f"{self.num_samples_points}")
        tb_file.add_text(f"mc_level",f"{self.mc_level}")
        tb_file.add_text(f"remove_outliers",f"{self.remove_outliers}")
        tb_file.add_text(f"depth_output_name",f"{self.depth_output_name}")
        tb_file.add_text(f"normal_method",f"{self.normal_method}")
        tb_file.add_text(f"output_file_name",f"{self.output_file_name}")
        tb_file.add_text(f"num_rays_per_batch",f"{self.num_rays_per_batch}")
        tb_file.add_text(f"bounding_box_min",f"{self.bounding_box_min}")
        tb_file.add_text(f"bounding_box_max",f"{self.bounding_box_max}")
        tb_file.add_text(f"ray_depth_Length",f"{self.ray_depth_length}")



        # Increase the batchsize to speed up the evaluation.
        ##pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        ## Finding density using marching cubes. density_fn used
        densities = density_sampler(
            pipeline=pipeline,
            num_samples=self.num_samples_mc,
            remove_outliers=self.remove_outliers,
            depth_output_name=self.depth_output_name,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
        )
        densities_flat = densities.reshape(-1)
        print(f"max densities = {np.amax(densities)}")
        print(f"average Denisites = {np.average(densities)}")
        print(f"Min Densities = {np.amin(densities)}")

        histogram =  np.histogram(densities_flat)
        tb_file.add_text(f"Density" ,f"Average: {np.average(densities)}, Max: {np.amax(densities)}, Min: {np.amin(densities)}")
        ##dense_histogram = display_histogram_of_densities(densities,self.output_dir,f"First_pass_{self.output_file_name[0:-4]}")

        dense_Avg = np.average(densities)
        torch.cuda.empty_cache()
        ##distance is 5% of the avg range of bounding box

        ##size of bb
        bb_size = tuple(map(lambda i, j: i - j, self.bounding_box_max, self.bounding_box_min))
        bb_avg = (bb_size[0] + bb_size[1] + bb_size[2]) / 3

        dist_along_normal = self.ray_depth_length
        print(f"ray length = {dist_along_normal}")

        device = o3d.core.Device("CUDA:0")
        dtype_f = o3d.core.float32
        dtype_i = o3d.core.int32

        verts, faces, normals, values = skimage.measure.marching_cubes(
            densities,
            allow_degenerate=False,
            level=dense_Avg,
            gradient_direction="ascent",
        )



        # convert properties to be compatible with cpu Triangle mesh(Has functions tesor does not)
        o3dVerts = o3d.utility.Vector3dVector(verts)
        o3dTris = o3d.utility.Vector3iVector(faces)
        o3dNorms = o3d.utility.Vector3dVector(normals)

        mesh = o3d.t.geometry.TriangleMesh(device)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3dVerts
        mesh.triangles = o3dTris
        mesh.vertex_normals = o3dNorms

        pcd = mesh.sample_points_uniformly(number_of_points=self.num_samples_points, use_triangle_normal=True)
        print(
            f"After points sampled from mesh: {torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()} gpu mem allocated"
        )
        torch.cuda.empty_cache()
        
        ##o3dvis.draw(pcd,show_ui=False)

        pcd_pos = np.asarray(pcd.points).astype(np.float32)  # N, 3
        pcd_norms = np.asarray(pcd.normals).astype(np.float32)  # N, 3

        pos_and_normals = torch.tensor(np.concatenate((pcd_pos, pcd_norms), -1))
        print(pos_and_normals)

        ##optimise from SAMURAI later
        refined_points = []
        refined_normals = []
        colours = []
        counter = 0
        chunk_size = 1000000 ##262144  # 65536 ##2^16
        ray_samples = 8
        samples_per_batch = chunk_size // ray_samples
        coloursCounter = 0
        coloursToUse = [
            [1.0, 0, 0],
            # [0, 1.0, 0],
            # [0, 0, 1.0],
            # [0.50, 0.50, 0],
            # [0.50, 0, 0.50],
            # [0, 0.50, 0.50],
            # [0.100, 0.100, 0.100],
            # [0.0, 0.0, 0.0],
        ]
        point_counter = 0

        densest_vals = torch.empty(0,device="cuda")

        for position_normal_sample in torch.tensor_split(
            input=pos_and_normals, sections=pos_and_normals.shape[0] // samples_per_batch, dim=0
        ):
            torch.cuda.empty_cache()
            s_time = time.time()
            position_sample = position_normal_sample[..., :3]
            normal_sample = position_normal_sample[..., 3:]

            ##direction from point along normal towards original point on mesh
            ##ray_direction = torch.tensor(math.safe_normalize((position_sample + normal_sample) - position_sample))
            ray_direction = torch.tensor(math.safe_normalize(normal_sample))

            # Ray origin at the extent of the distance along normal, stepping toward surface
            ray_origin = torch.tensor(position_sample + (-ray_direction * dist_along_normal))


 

            ##sample small area infront and behind original point
            sample_gap = torch.linspace(0.0, 2 * dist_along_normal, ray_samples)

            spaced_points = torch.empty(size=(ray_origin.shape[0], sample_gap.shape[0], ray_origin.shape[1]))
            ray_point_normals = torch.empty(size=(ray_origin.shape[0], sample_gap.shape[0], ray_origin.shape[1]))

            for i in range(0, sample_gap.size()[0]):
                spaced_points[:, i, :] = ray_origin + (ray_direction * sample_gap[i])

            for n in range(0, ray_origin.size()[0]):
                ray_point_normals[n, :, :] = normal_sample[n]

            # print(ray_origin)

            # print(spaced_points)
            # print(f"spaced points shape = {spaced_points.shape}")

            ##densities = pipeline.model.field.density_fn(spaced_points)


            # point_dens = torch.cat((spaced_points, densities), 2)
            # print(f"pointdens = {point_dens}")
            # print(f"densities = {densities}")

            ##densest_in_ray = densities.argmax(1)
            ##print(f"Before raysample declared: {torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()}")
            # Compute average of normals of each point sampled.
            ray_sam = RaySamples(
                frustums=Frustums(
                    origins=spaced_points.cuda(),
                    directions=torch.ones_like(ray_point_normals).cuda(),
                    starts=torch.zeros_like(spaced_points[..., :1]).cuda(),
                    ends=(torch.ones_like(spaced_points[..., :1]) * (dist_along_normal / ray_samples)).cuda(),
                    pixel_area=torch.ones_like(spaced_points[..., :1]).cuda(),
                ),
                camera_indices=torch.randint_like(spaced_points[..., :1], 150).cuda(),
            )
            print(f"Memory Usage: {torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()}")
            outputs = pipeline.model.field.forward(ray_sam, compute_normals=True)

            density_fn_densities = pipeline.model.field.density_fn(spaced_points)
            output_densities = outputs[FieldHeadNames.DENSITY]

            CONSOLE.print(f"Densities Range,avg - ({torch.min(density_fn_densities)} - {torch.max(density_fn_densities)}),{torch.mean(density_fn_densities)}")
            CONSOLE.print(f"Densities Range,avg - ({torch.min(output_densities)} - {torch.max(output_densities)}),{torch.mean(output_densities)}")
            densest_in_ray = output_densities.argmax(1)

            # print(f"after forward pass: {torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()}")

            normal_sample = outputs[FieldHeadNames.NORMALS]
            normal_sample = torch.tensor(normal_sample)
            normal_sample = torch.mean(normal_sample, 1)
            ## print(normal_sample)
            # originsv3d = o3d.utility.Vector3dVector(spaced_points.reshape([-1,3]))
            # colorv3d = o3d.utility.Vector3dVector(normal_sample.reshape([-1,3]).cpu().numpy())

            # debug_cloud = o3d.geometry.PointCloud(originsv3d)
            # debug_cloud.colors = colorv3d
            ##o3dvis.draw(debug_cloud)

            ###
            ###Testing densities retrieved from second pass
            torch.cat((densest_vals,output_densities.max(1)[0].reshape((-1))))              

            ###


            idx = 0
            colouridx = coloursCounter % len(coloursToUse)
            for d in densest_in_ray.cpu():

                if output_densities[idx, densest_in_ray[idx]] > 0.0:
                    refined_points.append(spaced_points[idx, d.cpu()])
                    refined_normals.append(normal_sample[idx])

                    point_counter += 1

                # ##testing. outputs all points sampled for some rays
                # if idx % 1000 == 0:
                #     for p in spaced_points[idx]:
                #         refined_points.append(torch.tensor([[p[0], p[1], p[2]]]))
                idx += 1

            coloursCounter += 1
            # print(f"after raysample deleted: {torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()}")
            e_time = time.time()

            print(f"Loop Time = {e_time - s_time}")
        ##ray_comp_histogram = display_histogram_of_densities(np.array(densest_vals),self.output_dir,f"DenseMax_refined_{self.output_file_name[0:-4]}")
        densest_vals_np = np.array(densest_vals.cpu())
        ##CONSOLE.print(f"Densities B After loop Range,avg - ({np.min(densest_vals_np)} - {np.max(densest_vals_np)}),{np.mean(densest_vals_np)}")
        
        
        print(f"pointCounter = {point_counter}")
        refined_points = torch.stack(refined_points).to(torch_device)
        refined_normals = torch.stack(refined_normals).to(torch_device)
        print("Torch tensors for points and normals stacked...")

        refined_points = refined_points.reshape((-1, 3))
        refined_normals = refined_normals.reshape((-1,3))
        # pointsv3d = o3d.utility.Vector3dVector(refined_points.cpu().numpy())
        # normalv3d = o3d.utility.Vector3dVector(refined_normals.cpu().numpy())
        # debug_cloud = o3d.geometry.PointCloud(pointsv3d)
        # debug_cloud.colors = normalv3d
        # o3dvis.draw(debug_cloud)

        # ray_sam = RaySamples(
        #     frustums=Frustums(
        #         origins=refined_points,
        #         directions=torch.ones_like(refined_points).to(torch_device),
        #         starts=torch.zeros_like(refined_points[..., :1]).to(torch_device),
        #         ends=torch.zeros_like(refined_points[..., :1]).to(torch_device),
        #         pixel_area=torch.ones_like(refined_points[..., :1]).to(torch_device),
        #     ),
        #     camera_indices=torch.zeros_like(refined_points[..., :1]).to(torch_device),
        # )

        # colours = torch.stack(colours)

        ##pipeline.model.field._sample_locations = refined_points
        # outputs = pipeline.model.field.forward(ray_sam, compute_normals=True)
        # print(outputs.keys())
        # refined_normals = outputs[FieldHeadNames.NORMALS]
        refined_normals = refined_normals.reshape((-1, 3))

        # print(refined_points)
        ref_pcd = o3d.geometry.PointCloud()
        ##vector must be transposed to create point cloud
        ref_verts = o3d.utility.Vector3dVector(refined_points.cpu().numpy())
        ref_norms = o3d.utility.Vector3dVector(refined_normals.cpu().detach().numpy())
        print("Verticies and normals of point cloud assigned to vecotr.")
        # ref_colours = o3d.utility.Vecto0r3dVector(colours.cpu().numpy())

        ref_pcd.points = ref_verts
        ##ref_pcd.normals = ref_norms
        ref_pcd.estimate_normals()
        ref_pcd.normalize_normals()
        print("Complex point cloud normals calculated")
        print(ref_pcd.points)
        print(ref_pcd.normals)
        ref_pcd.colors = pcd.normals
        ref_pcd.orient_normals_consistent_tangent_plane(100)

        ##o3dvis.draw(geometry=(ref_pcd))
        # ns-export samurai-mc --load-config outputs\data\tandt\ignatius\nerfacto\2023-03-21_171009/config.yml --output-dir exports/samurai/ --use-bounding-box True --bounding-box-min -0.2 -0.2 -0.25 --bounding-box-max 0.2 0.2 0.25 --num-samples-mc 100

        ##ns-export samurai-mc --load-config outputs\test-sphere\nerfacto\2023-04-04_165440/config.yml --output-dir exports/samurai/ --use-bounding-box True --bounding-box-min 0.013000000000000067 -0.24700000000000005 -0.15000000000000002 --bounding-box-max 0.3430000000000001 0.08299999999999998 0.18000000000000005 --num-samples-mc 250
        ## Construct mesh using Poisson Surface Reconstruction and removing bottom 98% Density Points
        for x in {8}:#{6,7,8,9}:
            for p in {0.01}:#{0.03,0.05,0.1,0.15,0.2,0.25,0.3}:
                CONSOLE.print(f"Densities Range,avg - ({np.min(densities)} - {np.max(densities)}),{np.average(densities)}")

                CONSOLE.print("Computing Mesh... this may take a while.")
                ##mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(ref_pcd, depth=x)
                vertices_to_remove = densities < 0.98## np.quantile(densities, p)
                mesh.remove_vertices_by_mask(vertices_to_remove)
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(ref_pcd, depth=x)
                print("\033[A\033[A")
                CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")
                
                ##outputs images of the mesh to tensorboard file
                render_mesh_to_tb(mesh,self.output_dir.__str__(),tb_file)

                if self.save_mesh:
                    ##Other programs for model veiwing read from 1. Python indexes from 0

                    path = self.output_dir.__str__() + f"\\{p}%removed_{x}mclevel{self.mc_level}" + self.output_file_name

                    o3d.io.write_triangle_mesh(path, mesh, print_progress=True)

        ##o3dvis.draw(mesh)

        colours = np.zeros_like(verts)

        CONSOLE.print(f"[bold green]:white_check_mark: Generated Marching Cube representation!!")

        if self.save_mesh:
            ##Other programs for model veiwing read from 1. Python indexes from 0
            facesReindex = faces + 1

            mcUtils.save_obj(verts, normals, facesReindex, self.output_dir, self.output_file_name)


@dataclass
class ExportMarchingTetTSDFMesh(Exporter):
    """
    Export a mesh using TSDF processing.
    """

    downscale_factor: int = 2
    """Downscale the images starting from the resolution used for training."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    resolution: Union[int, List[int]] = field(default_factory=lambda: [128, 128, 128])
    """Resolution of the TSDF volume or [x, y, z] resolutions individually."""
    batch_size: int = 10
    """How many depth images to integrate per batch."""
    use_bounding_box: bool = True
    """Whether to use a bounding box for the TSDF volume."""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    texture_method: Literal["tsdf", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'tsdf' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""
    loss_weights: Tuple[float,float,float,float] = (0.00001,0.000001,0.000001,0.000001)
    """DEBUG: Change the weights of the losses applied during training of the tsdf.\n surface, Normal Consist., Normal smooth., Normal orient."""
    batch_splits: int = 30
    """DEBUG: number of times the rays are split before propagation to sdf. altering can help with tsdf problems"""
    epochs: int = 3
    """DEBUG: number of times each ray is propagated through"""
    nerf_image_path: str= ""
    """path to prerendered data"""
    ray_limit: int = 77000000
    """Maximum amount of rays held in memory. Defaults to 77000000 which almost fills a 3090TI."""
    def main(self) -> None:
        """Export mesh"""

        torch.set_default_dtype(torch.float32)

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        print(self.output_dir)

        _, pipeline, _ = eval_setup(self.load_config)
        ##torch.set_anomaly_enabled(True,True)
        ssan = ssan_utils.export_ssan(
            pipeline,
            self.output_dir,
            self.downscale_factor,
            self.depth_output_name,
            self.rgb_output_name,
            self.resolution,
            self.batch_size,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            loss_weights=self.loss_weights,
            batch_splits= self.batch_splits,
            epochs= self.epochs,
            nerf_image_path=self.nerf_image_path
        )


        return 
        # possibly
        # texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the tsdf export
            mesh = get_mesh_from_filename(
                str(self.output_dir / "tsdf_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportCameraPoses(Exporter):
    """
    Export camera poses to a .json file.
    """

    def main(self) -> None:
        """Export camera poses"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)
        assert isinstance(pipeline, VanillaPipeline)
        train_frames, eval_frames = collect_camera_poses(pipeline)

        for file_name, frames in [("transforms_train.json", train_frames), ("transforms_eval.json", eval_frames)]:
            if len(frames) == 0:
                CONSOLE.print(f"[bold yellow]No frames found for {file_name}. Skipping.")
                continue

            output_file_path = os.path.join(self.output_dir, file_name)

            with open(output_file_path, "w", encoding="UTF-8") as f:
                json.dump(frames, f, indent=4)

            CONSOLE.print(f"[bold green]:white_check_mark: Saved poses to {output_file_path}")


@dataclass
class ExportGaussianSplat(Exporter):
    """
    Export 3D Gaussian Splatting model to a .ply
    """

    output_filename: str = "splat.ply"
    """Name of the output file."""
    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    ply_color_mode: Literal["sh_coeffs", "rgb"] = "sh_coeffs"
    """If "rgb", export colors as red/green/blue fields. Otherwise, export colors as
    spherical harmonics coefficients."""

    @staticmethod
    def write_ply(
        filename: str,
        count: int,
        map_to_tensors: typing.OrderedDict[str, np.ndarray],
    ):
        """
        Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
        Note: All float values will be converted to float32 for writing.

        Parameters:
        filename (str): The name of the file to write.
        count (int): The number of vertices to write.
        map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays of float or uint8 values.
            Each array should be 1-dimensional and of equal length matching 'count'. Arrays should not be empty.
        """

        # Ensure count matches the length of all tensors
        if not all(tensor.size == count for tensor in map_to_tensors.values()):
            raise ValueError("Count does not match the length of all tensors")

        # Type check for numpy arrays of type float or uint8 and non-empty
        if not all(
            isinstance(tensor, np.ndarray)
            and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
            and tensor.size > 0
            for tensor in map_to_tensors.values()
        ):
            raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

        with open(filename, "wb") as ply_file:
            nerfstudio_version = version("nerfstudio")
            # Write PLY header
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")
            ply_file.write(f"comment Generated by Nerstudio {nerfstudio_version}\n".encode())
            ply_file.write(b"comment Vertical Axis: z\n")
            ply_file.write(f"element vertex {count}\n".encode())

            # Write properties, in order due to OrderedDict
            for key, tensor in map_to_tensors.items():
                data_type = "float" if tensor.dtype.kind == "f" else "uchar"
                ply_file.write(f"property {data_type} {key}\n".encode())

            ply_file.write(b"end_header\n")

            # Write binary data
            # Note: If this is a performance bottleneck consider using numpy.hstack for efficiency improvement
            for i in range(count):
                for tensor in map_to_tensors.values():
                    value = tensor[i]
                    if tensor.dtype.kind == "f":
                        ply_file.write(np.float32(value).tobytes())
                    elif tensor.dtype == np.uint8:
                        ply_file.write(value.tobytes())

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config, test_mode="inference")

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model

        filename = self.output_dir / self.output_filename

        map_to_tensors = OrderedDict()

        with torch.no_grad():
            positions = model.means.cpu().numpy()
            count = positions.shape[0]
            n = count
            map_to_tensors["x"] = positions[:, 0]
            map_to_tensors["y"] = positions[:, 1]
            map_to_tensors["z"] = positions[:, 2]
            map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

            if self.ply_color_mode == "rgb":
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                colors = (colors * 255).astype(np.uint8)
                map_to_tensors["red"] = colors[:, 0]
                map_to_tensors["green"] = colors[:, 1]
                map_to_tensors["blue"] = colors[:, 2]
            elif self.ply_color_mode == "sh_coeffs":
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

            if model.config.sh_degree > 0:
                if self.ply_color_mode == "rgb":
                    CONSOLE.print(
                        "Warning: model has higher level of spherical harmonics, ignoring them and only export rgb."
                    )
                elif self.ply_color_mode == "sh_coeffs":
                    # transpose(1, 2) was needed to match the sh order in Inria version
                    shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                    shs_rest = shs_rest.reshape((n, -1))
                    for i in range(shs_rest.shape[-1]):
                        map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]

            map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

            scales = model.scales.data.cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]

            if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
                crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
                assert crop_obb is not None
                mask = crop_obb.within(torch.from_numpy(positions)).numpy()
                for k, t in map_to_tensors.items():
                    map_to_tensors[k] = map_to_tensors[k][mask]

                n = map_to_tensors["x"].shape[0]
                count = n

        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            n_before = np.sum(select)
            select = np.logical_and(select, np.isfinite(t).all(axis=-1))
            n_after = np.sum(select)
            if n_after < n_before:
                CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")
        nan_count = np.sum(select) - n

        # filter gaussians that have opacities < 1/255, because they are skipped in cuda rasterization
        low_opacity_gaussians = (map_to_tensors["opacity"]).squeeze(axis=-1) < -5.5373  # logit(1/255)
        lowopa_count = np.sum(low_opacity_gaussians)
        select[low_opacity_gaussians] = 0

        if np.sum(select) < n:
            CONSOLE.print(
                f"{nan_count} Gaussians have NaN/Inf and {lowopa_count} have low opacity, only export {np.sum(select)}/{n}"
            )
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select]
            count = np.sum(select)

        ExportGaussianSplat.write_ply(str(filename), count, map_to_tensors)


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
        Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
        Annotated[ExportPoissonMesh, tyro.conf.subcommand(name="poisson")],
        Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
        Annotated[ExportSamuraiMarchingCubes, tyro.conf.subcommand(name="samurai-mc")],
        Annotated[ExportMarchingTetTSDFMesh, tyro.conf.subcommand(name="Marching-tet")],
        Annotated[ExportCameraPoses, tyro.conf.subcommand(name="cameras")],
        Annotated[ExportGaussianSplat, tyro.conf.subcommand(name="gaussian-splat")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
