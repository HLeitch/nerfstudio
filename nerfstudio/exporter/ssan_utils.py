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
    
    surface_mlp = 

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

        # TODO: move to device

        return TSDF(voxel_coords, values, weights,normal_values,normal_weights, colors, voxel_size, origin)

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

    def integrate_tsdf(
        self,
        c2w: TensorType["batch", 4, 4],
        K: TensorType["batch", 3, 3],
        depth_images: TensorType["batch", 1, "height", "width"],
        color_images: Optional[TensorType["batch", 3, "height", "width"]] = None,
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

        # make voxel_coords homogeneous
        voxel_world_coords = self.voxel_coords.view(3, -1)
        voxel_world_coords = torch.cat(
            [voxel_world_coords, torch.ones(1, voxel_world_coords.shape[1], device=self.device)], dim=0
        )
        voxel_world_coords = voxel_world_coords.unsqueeze(0)  # [1, 4, N]
        voxel_world_coords = voxel_world_coords.expand(batch_size, *voxel_world_coords.shape[1:])  # [batch, 4, N]

        voxel_cam_coords = torch.bmm(torch.inverse(c2w), voxel_world_coords)  # [batch, 4, N]

        # flip the z axis
        voxel_cam_coords[:, 2, :] = -voxel_cam_coords[:, 2, :]
        # flip the y axis
        voxel_cam_coords[:, 1, :] = -voxel_cam_coords[:, 1, :]

        # we need the distance of the point to the camera, not the z coordinate
        voxel_depth = torch.sqrt(torch.sum(voxel_cam_coords[:, :3, :] ** 2, dim=-2, keepdim=True))  # [batch, 1, N]

        voxel_cam_coords_z = voxel_cam_coords[:, 2:3, :]
        voxel_cam_points = torch.bmm(K, voxel_cam_coords[:, 0:3, :] / voxel_cam_coords_z)  # [batch, 3, N]
        voxel_pixel_coords = voxel_cam_points[:, :2, :]  # [batch, 2, N]

        # Sample the depth images with grid sample...

        grid = voxel_pixel_coords.permute(0, 2, 1)  # [batch, N, 2]
        # normalize grid to [-1, 1]
        grid = 2.0 * grid / image_size.view(1, 1, 2) - 1.0  # [batch, N, 2]
        grid = grid[:, None]  # [batch, 1, N, 2]
        # depth
        sampled_depth = F.grid_sample(
            input=depth_images, grid=grid, mode="nearest", padding_mode="zeros", align_corners=False
        )  # [batch, N, 1]
        sampled_depth = sampled_depth.squeeze(2)  # [batch, 1, N]
        # colors
        if color_images is not None:
            sampled_colors = F.grid_sample(
                input=color_images, grid=grid, mode="nearest", padding_mode="zeros", align_corners=False
            )  # [batch, N, 3]
            sampled_colors = sampled_colors.squeeze(2)  # [batch, 3, N]


        dist = sampled_depth - voxel_depth  # [batch, 1, N]
        tsdf_values = torch.clamp(dist / self.truncation, min=-1.0, max=1.0)  # [batch, 1, N]

        valid_points = (voxel_depth > 0) & (sampled_depth > 0) & (dist > -self.truncation)  # [batch, 1, N]

        # Sequentially update the TSDF...

        for i in range(batch_size):
            valid_points_i = valid_points[i]
            valid_points_i_shape = valid_points_i.view(*shape)  # [xdim, ydim, zdim]

            # the old values
            old_tsdf_values_i = self.values[valid_points_i_shape]
            old_weights_i = self.weights[valid_points_i_shape]

            # the new values
            # TODO: let the new weight be configurable
            new_tsdf_values_i = tsdf_values[i][valid_points_i]
            new_weights_i = 1.0

            total_weights = old_weights_i + new_weights_i

            self.values[valid_points_i_shape] = (
                old_tsdf_values_i * old_weights_i + new_tsdf_values_i * new_weights_i
            ) / total_weights
            self.weights[valid_points_i_shape] = torch.clamp(total_weights, max=1.0)

            if color_images is not None:
                old_colors_i = self.colors[valid_points_i_shape]  # [M, 3]
                new_colors_i = sampled_colors[i][:, valid_points_i.squeeze(0)].permute(1, 0)  # [M, 3]
                self.colors[valid_points_i_shape] = (
                    old_colors_i * old_weights_i[:, None] + new_colors_i * new_weights_i
                ) / total_weights[:, None]


    def integrate_tri_tsdf(
        self,
        c2w: TensorType["batch", 4, 4],
        K: TensorType["batch", 3, 3],
        depth_images: TensorType["batch", 1, "height", "width"],
        depth_images_outside: TensorType["batch", 1, "height", "width"],
        depth_images_inside: TensorType["batch", 1, "height", "width"],
        ##ray_origins: TensorType["batch", 3, "height", "width"],
        surface_normals: TensorType["batch", 3, "height", "width"],
        normal_samples: TensorType["batch", 3, "height", "width"],
        normal_regularity: TensorType["batch", 1, "height", "width"],
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

        # make voxel_coords homogeneous
        voxel_world_coords = self.voxel_coords.view(3, -1)
        voxel_world_coords = torch.cat(
            [voxel_world_coords, torch.ones(1, voxel_world_coords.shape[1], device=self.device)], dim=0
        )
        voxel_world_coords = voxel_world_coords.unsqueeze(0)  # [1, 4, N]
        voxel_world_coords = voxel_world_coords.expand(batch_size, *voxel_world_coords.shape[1:])  # [batch, 4, N]

        voxel_cam_coords = torch.bmm(torch.inverse(c2w), voxel_world_coords)  # [batch, 4, N]

        # flip the z axis
        voxel_cam_coords[:, 2, :] = -voxel_cam_coords[:, 2, :]
        # flip the y axis
        voxel_cam_coords[:, 1, :] = -voxel_cam_coords[:, 1, :]

        # we need the distance of the point to the camera, not the z coordinate
        voxel_depth = torch.sqrt(torch.sum(voxel_cam_coords[:, :3, :] ** 2, dim=-2, keepdim=True))  # [batch, 1, N]

        voxel_cam_coords_z = voxel_cam_coords[:, 2:3, :]
        voxel_cam_points = torch.bmm(K, voxel_cam_coords[:, 0:3, :] / voxel_cam_coords_z)  # [batch, 3, N]
        voxel_pixel_coords = voxel_cam_points[:, :2, :]  # [batch, 2, N]
        del(voxel_cam_coords_z,voxel_cam_points,voxel_cam_coords,voxel_world_coords)

        # Sample the depth images with grid sample...

        grid = voxel_pixel_coords.permute(0, 2, 1)  # [batch, N, 2]
        # normalize grid to [-1, 1]
        grid = 2.0 * grid / image_size.view(1, 1, 2) - 1.0  # [batch, N, 2]
        grid = grid[:, None]  # [batch, 1, N, 2]
        # depth surface
        sampled_depth = F.grid_sample(
            input=depth_images, grid=grid, mode="nearest", padding_mode="zeros", align_corners=False
        )  # [batch, N, 1]
        sampled_depth = sampled_depth.squeeze(2)  # [batch, 1, N]
        print(F"sampled depth size: {sampled_depth.shape}")
        # depth Outside
        sampled_depth_16 = F.grid_sample(
            input=depth_images_outside, grid=grid, mode="nearest", padding_mode="zeros", align_corners=False
        )  # [batch, N, 1]
        sampled_depth_16 = sampled_depth_16.squeeze(2)  # [batch, 1, N]

        # depth Inside 
        sampled_depth_84 = F.grid_sample(
            input=depth_images_inside, grid=grid, mode="nearest", padding_mode="zeros", align_corners=False
        )  # [batch, N, 1]
        sampled_depth_84 = sampled_depth_84.squeeze(2)  # [batch, 1, N]


        del(depth_images,depth_images_outside,depth_images_inside)

        # sampled_origins = F.grid_sample(
        #     input=ray_origins, grid=grid, mode="nearest", padding_mode="zeros", align_corners=False
        # )  # [batch, N, 1]
        # sampled_origins = sampled_origins.squeeze(2)  # [batch, 1, N]

        ## normals
        surface_normals_grid = F.grid_sample(input=surface_normals, grid=grid,mode="nearest",padding_mode="zeros",align_corners=False
            ) # [batch, N, 3])
        surface_normals_grid = surface_normals_grid.squeeze(2)

        normal_samples_grid = F.grid_sample(
            input=normal_samples, grid=grid,mode="nearest",padding_mode="zeros",align_corners=False
            ) # [batch, N, 3]
        normal_samples_grid = normal_samples_grid.squeeze(2)

        normal_regularity_grid = F.grid_sample(
            input=normal_regularity, grid=grid,mode="nearest",padding_mode="zeros",align_corners=False
            ) # [batch, N, 3]
        normal_regularity_grid = normal_regularity_grid.squeeze(2)

        # colors
        # if color_images is not None:
        #     sampled_colors = F.grid_sample(
        #         input=color_images, grid=grid, mode="nearest", padding_mode="zeros", align_corners=False
        #     )  # [batch, N, 3]
        #     sampled_colors = sampled_colors.squeeze(2)  # [batch, 3, N]

        surface_dist = sampled_depth - voxel_depth  # [batch, 1, N]
        # outside_dist = sampled_depth_16 - voxel_depth
        # inside_dist = sampled_depth_84 - voxel_depth

        hyperparameter = 1
        print(surface_dist)
        tsdf_values_surface = torch.clamp(surface_dist / self.truncation, min=-1.0, max=1.0)  # [batch, 1, N]
        tsdf_values_outside = torch.clamp(torch.Tensor((surface_dist / self.truncation)), min=-1.0, max=1.0) - hyperparameter  # [batch, 1, N]
        tsdf_values_inside = torch.clamp(torch.Tensor((surface_dist / self.truncation)), min=-1.0, max=1.0) + hyperparameter  # [batch, 1, N]

        # print(f"tsdf_values_outside: {tsdf_values_outside}")
        # print(f"tsdf_values_inside: {tsdf_values_inside}")
        

        tsdf_values = tsdf_values_surface##(tsdf_values_outside + tsdf_values_surface + tsdf_values_inside)/3

        valid_points = (voxel_depth > 0) & (sampled_depth > 0) & (surface_dist > -self.truncation)  # [batch, 1, N]
        
        # Sequentially update the TSDF...
        for i in range(batch_size):
                
            valid_points_i = valid_points[i]
            valid_points_i_shape = valid_points_i.view(*shape)  # [xdim, ydim, zdim]
            

            # the old values
            old_tsdf_values_i = self.values[valid_points_i_shape]
            old_weights_i = self.weights[valid_points_i_shape]


            old_normal_values_i = self.normal_values[valid_points_i_shape]
            old_normal_weights_i = self.normal_weights[valid_points_i_shape]

            # the new values
            # TODO: let the new weight be configurable

            # Rescale to limits of [-0.1,0.1]
            new_tsdf_values_surface_i = tsdf_values_surface[i][valid_points_i] 
            new_tsdf_values_outside_i = tsdf_values_outside[i][valid_points_i]
            new_tsdf_values_inside_i = tsdf_values_inside[i][valid_points_i]

            new_normal_values = surface_normals_grid[i][:, valid_points_i.squeeze(0)].permute(1, 0)  # [M, 3]
            print(f"normal regularity {normal_regularity.shape}")
            normal_regularity_i = normal_regularity_grid[i][valid_points_i]
            normal_samples_i = normal_samples_grid[i][:, valid_points_i.squeeze(0)].permute(1, 0)  # [M, 3]

            
            # print(f"Inside: {new_tsdf_values_inside_i}")
            #print(f"SurfaceAll: {tsdf_values_surface.shape}")
            print(f"Surface: {valid_points_i}")
            # print(f"Outside: {new_tsdf_values_outside_i}")


            ##To give a magnitiude similar to NeRFMeshing paper, we muliply loss by 0.1. This also means we
            ## can keep the weight clamps at 1.
            surface_loss = (((new_tsdf_values_outside_i)**2)+
                            (new_tsdf_values_surface_i**2)+
                            ((new_tsdf_values_inside_i)**2))
            
            #print(f"Surface Loss min: {surface_loss.min()}. Surface loss count: {surface_loss.numel()}. Loss per value: {surface_loss.sum()/surface_loss.numel()}")
            print(f"Surface Loss elementwise: {surface_loss}")

            ## Theoretical maximum loss is 5 when hyperparameter and range is 1 and -1 -> 1. If the loss is greater or equal to 5,
            ## no weight is added. IMPLIMENT NEXT 
            new_weights_i =(1.0/((0.1+surface_loss))) ##torch.abs((5.01-surface_loss)/5)##

            total_weights = old_weights_i + new_weights_i
            print(f"Old Weights: {total_weights}")

            self.values[valid_points_i_shape] = (
                old_tsdf_values_i * old_weights_i + new_tsdf_values_surface_i * new_weights_i
            ) / total_weights
            self.weights[valid_points_i_shape] = torch.clamp(total_weights, max=1.0)

            ##Normal Weight Calculation
            normal_loss = 10 - normal_regularity_grid
            normal_loss = normal_loss**2
            ##prevents weight from exceeding 1
            normal_weight = (1.0/(1+normal_loss))
            del(normal_loss)

            ##Normal regularization
            regularization_loss = torch.tensor(torch.norm((new_normal_values - old_normal_values_i),dim=1)).pow(2)
            print(f"regularization loss = {regularization_loss}")

            assert False

            # if color_images is not None:
            #     old_colors_i = self.colors[valid_points_i_shape]  # [M, 3]
            #     new_colors_i = sampled_colors[i][:, valid_points_i.squeeze(0)].permute(1, 0)  # [M, 3]
            #     self.colors[valid_points_i_shape] = (
            #         old_colors_i * old_weights_i[:, None] + new_colors_i * new_weights_i[:, None]
            #     ) / total_weights[:, None]