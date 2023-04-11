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
import skimage.measure
import torch
import tyro
from rich.console import Console
from typing_extensions import Annotated, Literal

import nerfstudio.cameras.cameras as nscam
import nerfstudio.exporter.marching_cubes_utils as mcUtils
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import (
    collect_camera_poses,
    density_sampler,
    generate_point_cloud,
    get_mesh_from_filename,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils import math as math
from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""


@dataclass
class ExportPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    estimate_normals: bool = False
    """Estimate normals for the point cloud."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Maximum of the bounding box, used if use_bounding_box is True."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""

    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _ = eval_setup(self.load_config)

        # Increase the batchsize to speed up the evaluation.
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            estimate_normals=self.estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=None,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            std_ratio=self.std_ratio,
        )
        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")
        o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
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

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _ = eval_setup(self.load_config)

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

    def validate_pipeline(self, pipeline: Pipeline) -> None:
        """Check that the pipeline is valid for this exporter."""
        if self.normal_method == "model_output":
            CONSOLE.print("Checking that the pipeline has a normal output.")
            origins = torch.zeros((1, 3), device=pipeline.device)
            directions = torch.ones_like(origins)
            pixel_area = torch.ones_like(origins[..., :1])
            camera_indices = torch.zeros_like(origins[..., :1])
            ray_bundle = RayBundle(
                origins=origins, directions=directions, pixel_area=pixel_area, camera_indices=camera_indices
            )
            outputs = pipeline.model(ray_bundle)
            if self.normal_output_name not in outputs:
                CONSOLE.print(
                    f"[bold yellow]Warning: Normal output '{self.normal_output_name}' not found in pipeline outputs."
                )
                CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
                CONSOLE.print(
                    "[bold yellow]Warning: Please train a model with normals "
                    "(e.g., nerfacto with predicted normals turned on)."
                )
                CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
                CONSOLE.print("[bold yellow]Exiting early.")
                sys.exit(1)

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _ = eval_setup(self.load_config)
        self.validate_pipeline(pipeline)

        # Increase the batchsize to speed up the evaluation.
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
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

        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _, pipeline, _ = eval_setup(self.load_config)

        self.validate_pipeline(pipeline)

        # Increase the batchsize to speed up the evaluation.
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        densities = density_sampler(
            pipeline=pipeline,
            num_samples=self.num_samples_mc,
            remove_outliers=self.remove_outliers,
            depth_output_name=self.depth_output_name,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
        )
        torch.cuda.empty_cache()

        ##distance is 5% of the avg range of bounding box

        ##size of bb
        bb_size = tuple(map(lambda i, j: i - j, self.bounding_box_max, self.bounding_box_min))
        bb_avg = (bb_size[0] + bb_size[1] + bb_size[2]) / 3

        dist_along_normal = 0.5  # bb_avg * 0.1
        print(f"ray length = {dist_along_normal}")

        device = o3d.core.Device("CUDA:0")
        dtype_f = o3d.core.float32
        dtype_i = o3d.core.int32

        verts, faces, normals, values = skimage.measure.marching_cubes(
            densities,
            allow_degenerate=False,
            level=self.mc_level,
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

        pcd = mesh.sample_points_uniformly(number_of_points=1000000, use_triangle_normal=True)
        print(
            f"After points sampled from mesh: {torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()} gpu mem allocated"
        )
        torch.cuda.empty_cache()
        o3dvis.draw(pcd)
        pcd_pos = np.asarray(pcd.points).astype(np.float32)  # N, 3
        pcd_norms = np.asarray(pcd.normals).astype(np.float32)  # N, 3

        pos_and_normals = torch.tensor(np.concatenate((pcd_pos, pcd_norms), -1))
        print(pos_and_normals)

        ##optimise from SAMURAI later
        refined_points = []
        refined_normals = []
        colours = []
        counter = 0
        chunk_size = 262144  # 65536 ##2^16
        ray_samples = 16
        samples_per_batch = chunk_size // ray_samples
        coloursCounter = 0
        coloursToUse = [
            [1.0, 0, 0],
            [0, 1.0, 0],
            [0, 0, 1.0],
            [0.50, 0.50, 0],
            [0.50, 0, 0.50],
            [0, 0.50, 0.50],
            [0.100, 0.100, 0.100],
            [0.0, 0.0, 0.0],
        ]
        point_counter = 0

        for position_normal_sample in torch.tensor_split(
            input=pos_and_normals, sections=pos_and_normals.shape[0] // samples_per_batch, dim=0
        ):
            torch.cuda.empty_cache()
            s_time = time.time()
            position_sample = position_normal_sample[..., :3]
            normal_sample = position_normal_sample[..., 3:]

            ##direction from point along normal towards original point on mesh
            ray_direction = torch.tensor(math.safe_normalize((position_sample + normal_sample) - position_sample))

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

            densities = pipeline.model.field.density_fn(spaced_points)

            # point_dens = torch.cat((spaced_points, densities), 2)
            # print(f"pointdens = {point_dens}")
            # print(f"densities = {densities}")

            densest_in_ray = densities.argmax(1)
            print(f"Before raysample declared: {torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()}")
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
            # print(f"Before raysample deleted: {torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()}")
            outputs = pipeline.model.field.forward(ray_sam, compute_normals=True)
            # print(f"after forward pass: {torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()}")

            normal_sample = outputs[FieldHeadNames.NORMALS]
            ##normal_sample = torch.mean(normal_sample, 1)
            ## print(normal_sample)
            idx = 0
            colouridx = coloursCounter % len(coloursToUse)
            for d in densest_in_ray:

                if densities[idx, densest_in_ray[idx]] > 0.0:
                    refined_points.append(spaced_points[idx, d])
                    refined_normals.append(normal_sample[idx, d])

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

        print(f"pointCounter = {point_counter}")
        refined_points = torch.stack(refined_points).to(torch_device)
        refined_normals = torch.stack(refined_normals).to(torch_device)

        refined_points = refined_points.reshape((-1, 3))

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
        # ref_colours = o3d.utility.Vecto0r3dVector(colours.cpu().numpy())

        ref_pcd.points = ref_verts
        ref_pcd.normals = ref_norms
        # ref_pcd.estimate_normals()
        print(ref_pcd.points)
        ref_pcd.colors = ref_norms
        o3dvis.draw(geometry=(ref_pcd))
        # ns-export samurai-mc --load-config outputs\data\tandt\ignatius\nerfacto\2023-03-21_171009/config.yml --output-dir exports/samurai/ --use-bounding-box True --bounding-box-min -0.2 -0.2 -0.25 --bounding-box-max 0.2 0.2 0.25 --num-samples-mc 100

        ##ns-export samurai-mc --load-config outputs\test-sphere\nerfacto\2023-04-04_163415/config.yml --output-dir exports/samurai/ --use-bounding-box True --bounding-box-min 0.07500000000000001 -0.225 -0.075 --bounding-box-max 0.325 0.024999999999999994 0.175 --num-samples-mc 250

        for x in {9}:
            CONSOLE.print("Computing Mesh... this may take a while.")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(ref_pcd, depth=x)
            vertices_to_remove = densities < np.quantile(densities, 0.3)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            print("\033[A\033[A")
            CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

            if self.save_mesh:
                ##Other programs for model veiwing read from 1. Python indexes from 0

                path = self.output_dir.__str__() + "\\" + f"maxDepthNorms 0.3removed" + self.output_file_name

                o3d.io.write_triangle_mesh(path, mesh, print_progress=True)

        ##o3dvis.draw(mesh)

        colours = np.zeros_like(verts)

        CONSOLE.print(f"[bold green]:white_check_mark: Generated Marching Cube representation!!")

        # if self.save_mesh:
        #     ##Other programs for model veiwing read from 1. Python indexes from 0
        #     facesReindex = faces + 1

        #     mcUtils.save_obj(verts, normals, facesReindex, self.output_dir, self.output_file_name)


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

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _ = eval_setup(self.load_config)

        tsdf_utils.export_marching_tet(
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
class ExportCameraPoses(Exporter):
    """
    Export camera poses to a .json file.
    """

    def main(self) -> None:
        """Export camera poses"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _ = eval_setup(self.load_config)
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


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
        Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
        Annotated[ExportPoissonMesh, tyro.conf.subcommand(name="poisson")],
        Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
        Annotated[ExportSamuraiMarchingCubes, tyro.conf.subcommand(name="samurai-mc")],
        Annotated[ExportMarchingTetTSDFMesh, tyro.conf.subcommand(name="Marching-tet")],
        Annotated[ExportCameraPoses, tyro.conf.subcommand(name="cameras")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa
