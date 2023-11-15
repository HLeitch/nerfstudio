
import os as os

import matplotlib.pyplot as plt
import numpy as np
import point_cloud_utils as pcu
import pytorch3d as torch3d
import pytorch3d.io as torchio
import pytorch3d.ops as torchops
from pytorch3d.structures import Pointclouds
import torch as torch
import pandas
import time

def IterationTest(file_a, file_b, num_points):
    print(f"Test for quality of relocation for {num_points} points")
    time_start = time.time()
    
    # Generate two random point sets
    a = pcu.load_mesh_v(file_a)
    b = pcu.load_mesh_v(file_b)
    
    
    
    
    a_tensor_full = torch.tensor(a,dtype=torch.float)
    b_tensor_full = torch.tensor(b,dtype=torch.float)

    
    randomPointsA = torch.randperm(a_tensor_full.shape[0])
    randomPointsB = torch.randperm(b_tensor_full.shape[0])
    
    ##randomly select a number of points 
    a_tensor = a_tensor_full[randomPointsA]
    b_tensor = b_tensor_full[randomPointsB]
    
    a_tensor = a_tensor[:num_points][None,:,:]
    b_tensor = b_tensor[:num_points][None,:,:]
    
    print(f"{a_tensor.shape}")
    #result = 
    
    # %%
    
    # Moving both sets of points to the origin.
    # avg_a = np.mean(a_tensor.numpy(),0)
    # avg_b = np.mean(b_tensor.numpy(),0)
    
    # print(f"{avg_a}, {avg_b}")
    # a = avg_a
    # b = avg_b
    
    # print(f"{np.mean(a,0)}, {np.mean(b,0)}")
    fig = plt.figure()  
    ax = fig.add_subplot(111,projection="3d")
    
    ax.scatter(a_tensor[0,:,0],a_tensor[0,:,1],a_tensor[0,:,2],marker=".")
    ax.scatter(b_tensor[0,:,0],b_tensor[0,:,1],b_tensor[0,:,2],marker=".",color="r")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    
    plt.show()
    #%%
    
    a_pc = Pointclouds(a_tensor)
    b_pc = Pointclouds(b_tensor)
    a_full_pc = Pointclouds(a_tensor_full[None,:,:])
    b_full_pc = Pointclouds(b_tensor_full[None,:,:])
    #%%
    
    output = torchops.iterative_closest_point(a_pc, b_pc,verbose=True,max_iterations=500, relative_rmse_thr=1e-7)
    rmse = output.rmse
    transformed_a = output.Xt
    RTs = output.RTs
    time_taken = time.time() - time_start
    print(time_taken)
    
    iterations_taken = output.t_history.__len__()
    print(iterations_taken)
    
    ## reduced a points only
    small_a = transformed_a.points_padded()
    
    ##Applying similarity transform(taken from ICP code from function above.)
    a_full_pc = RTs.s[:,None,None] * torch.bmm(a_full_pc.points_padded(), RTs.R) + RTs.T[:,None,:]
    
    #%%
    print(f"converted: {output.converged}, rmse{output.rmse}")
    print(f"simTransform {output.RTs}")
    
    #%%
    print(f"xT = {output.Xt}")
    ##transformed = output.Xt.points_list()
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection="3d")
    
    # ax.scatter(a_full_pc[:,0],a_full_pc[:,1],a_full_pc[:,2],marker=".")
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    
    
    # plt.show()
    
    # %%
    # Compute one-sided squared Hausdorff distances
    ## Reduced point cloud
    reduced_pc_comparison = pcu.one_sided_hausdorff_distance(small_a.cpu().numpy()[0], b_tensor.cpu().numpy()[0])
    
    ## full point clouds
    full_pc_comparison = pcu.one_sided_hausdorff_distance(a_full_pc[0].cpu().numpy(), b_full_pc.points_padded().cpu().numpy()[0])
    
    print(f"Reduced Hausdorff Reduced Data: {reduced_pc_comparison}")
    print(f"Directional Hausdorff Full Data: {full_pc_comparison}")
    
    
    # %%
    
    # Take a max of the one sided squared  distances to get the two sided Hausdorff distance
    general_hausdorff = pcu.hausdorff_distance(a_full_pc[0].cpu().numpy(), b_full_pc.points_padded().cpu().numpy()[0])
    
    # # Find the index pairs of the two points with maximum shortest distancce
    # hausdorff_b_to_a, idx_b, idx_a = pcu.one_sided_hausdorff_distance(b, transformed_a, return_index=True)
    # assert np.abs(np.sum((a[idx_a] - b[idx_b])**2) - hausdorff_b_to_a**2) < 1e-5, "These values should be almost equal"
    # print(f"Hausdorff shortest: {hausdorff_a_to_b}")
    
    
    # # Find the index pairs of the two points with maximum shortest distancce
    # hausdorff_dist, idx_b, idx_a = pcu.hausdorff_distance(b, transformed_a, return_index=True)
    # assert np.abs(np.sum((a[idx_a] - b[idx_b])**2) - hausdorff_dist**2) < 1e-5, "These values should be almost equal"  
    # print(f"Hausdorff max: {hausdorff_a_to_b}")
    
    ##Save rotated point cloud
    pcu.save_mesh_v((file_a[:-4]+f"matched{num_points}.obj"), a_full_pc[0].cpu().numpy())

    return rmse.cpu().numpy(), reduced_pc_comparison, full_pc_comparison, general_hausdorff, time_taken, iterations_taken
#%%
results = pandas.DataFrame([],columns=['num_points','rmse','ICP_hausdorff','Full_1D_hausdorff','Full_2D_hausdorff','time_taken','iterations'])

for num_points in [100, 200, 500, 750, 1000, 1500, 2000, 5000, 7500, 10000, 15000, 20000]:
    rmse, reduced_pc_comparison, full_pc_comparison, general_hausdorff, time_taken, iterations_taken = IterationTest( "./data/nerf-synthetic/lego/lego_pointcloud_rotated.obj", "./data/nerf-synthetic/lego/lego_pointcloud.obj", num_points)
    ##IterationTest( "./data/nerf-synthetic/lego/lego_pointcloud_rotated.obj", "./data/nerf-synthetic/lego/lego_pointcloud.obj", num_points)
    ##IterationTest("./data/tandt/Ignatius/Ignatius_z_rot.obj","./data/tandt/Ignatius/ignatius_base.obj",  num_points) ##IterationTest( "./data/tandt/Caterpillar/Caterpillar_shifted.obj","./data/tandt/Caterpillar/Caterpillar_base.obj", num_points)
    
    new_row = {'num_points':num_points,'rmse':rmse, 'ICP_hausdorff':reduced_pc_comparison, 'Full_1D_hausdorff':full_pc_comparison, 'Full_2D_hausdorff':general_hausdorff, 'time_taken': time_taken, 'iterations': iterations_taken}
    
    results = results.append(new_row,ignore_index=True)
print(results.to_string())
        
