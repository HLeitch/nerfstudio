
import os as os
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas
import point_cloud_utils as pcu
import pytorch3d as torch3d
import pytorch3d.io as torchio
import pytorch3d.ops as torchops
import torch as torch
from pytorch3d.structures import Pointclouds


def IterationTest(file_a, file_b, num_points):
    print(f"Test for quality of relocation for {num_points} points")
    time_start = time.time()
    
    # Generate two random point sets
    a = pcu.load_mesh_v(file_a)
    b = pcu.load_mesh_v(file_b)
    
        ##Moving both sets of points to the origin.
    avg_a = np.mean(a,0)
    avg_b = np.mean(b,0)
    
    print(f"{avg_a}, {avg_b}")
    a = a - avg_a
    b = b - avg_b

    # a = 100*a
    # b = 100*b
    
    # print(f"{np.mean(a,0)}, {np.mean(b,0)}")
    
    
    a_tensor_full = torch.tensor(a,dtype=torch.float)
    b_tensor_full = torch.tensor(b,dtype=torch.float)

    
    randomPointsA = torch.randperm(a_tensor_full.shape[0])
    randomPointsB = torch.randperm(b_tensor_full.shape[0])


    print("Sampling futhest points Test pcd")
    pointsA = torchops.sample_farthest_points(a_tensor_full[None, :, :],K=num_points)[0]
    print("Sampling futhest points Ground Truth pcd")
    pointsB = torchops.sample_farthest_points(b_tensor_full[None, :, :],K=num_points)[0]


    ##randomly select a number of points 
    a_tensor = a_tensor_full[randomPointsA]
    b_tensor = b_tensor_full[randomPointsB]
    
    a_tensor = a_tensor[:100000][None,:,:]
    b_tensor = b_tensor[:100000][None,:,:]

    ##just for testing correspondance
    # pointsA = torchops.sample_farthest_points(a_tensor,K=num_points)[0]
    # pointsB = torchops.sample_farthest_points(b_tensor, K=num_points)[0]

    ##sampling comparison scatter plot

    # fig = plt.figure()  
    # ax = fig.add_subplot(111,projection="3d")
    
    # furthest_points = ax.scatter(pointsA[0,:,0]+0.05,pointsA[0,:,1],pointsA[0,:,2],marker=".",label="Rotated Points")
    # random_points = ax.scatter(pointsB[0,:,0],pointsB[0,:,1],pointsB[0,:,2],marker=".",color="r",label="Base Points")
    # ax.legend(handles=[furthest_points,random_points])
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()
    # print(f"{a_tensor.shape}")
    # #result = 
    
    # %%
    

    # fig = plt.figure()  
    # ax = fig.add_subplot(111,projection="3d")

    
    # ax.scatter(a_tensor[0,:,0],a_tensor[0,:,1],a_tensor[0,:,2],marker=".")
    # ax.scatter(b_tensor[0,:,0],b_tensor[0,:,1],b_tensor[0,:,2],marker=".",color="r")
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    
    
    ##plt.show()
    #%%
    
    a_pc = Pointclouds(pointsA)
    b_pc = Pointclouds(pointsB)
    a_full_pc = Pointclouds(a_tensor_full[None,:,:])
    b_full_pc = Pointclouds(b_tensor_full[None,:,:])
    #%%
    
    output = torchops.iterative_closest_point(a_pc, b_pc,verbose=True,max_iterations=500, relative_rmse_thr=1e-9)
    ##output = torchops.corresponding_points_alignment(a_pc,b_pc,allow_reflection=True)

    rmse = output.rmse
    transformed_a = output.Xt
    RTs = output.RTs
    time_taken = time.time() - time_start
    print(f"time taken {time_taken}")

    ##iterations_taken = output.t_history.__len__()
    ##print(f"iterations_taken {iterations_taken}")
    
    ## reduced a points only
    small_a_pc = RTs.s[:,None,None] * torch.bmm(a_pc.points_padded(), RTs.R) + RTs.T[:,None,:]
    
    # fig = plt.figure()  
    # ax = fig.add_subplot(111,projection="3d")
    
    # furthest_points = ax.scatter(small_a_pc[0,:,0],small_a_pc[0,:,1],small_a_pc[0,:,2],marker=".",label="Rotated Points")
    # random_points = ax.scatter(pointsB[0,:,0],pointsB[0,:,1],pointsB[0,:,2],marker=".",color="r",label="Base Points")
    # ax.legend(handles=[furthest_points,random_points])
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

    ##Applying similarity transform(taken from ICP code from function above.)
    a_full_pc = RTs.s[:,None,None] * torch.bmm(a_full_pc.points_padded(), RTs.R) + RTs.T[:,None,:]
    
    #%%
    print(f"converted: {output.converged}, rmse{output.rmse}")
    print(f"simTransform {output.RTs}")
    
    #%%
    ##print(f"xT = {output.Xt}")
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
    reduced_pc_comparison = 0##pcu.one_sided_hausdorff_distance(small_a_pc.cpu().numpy()[0], b_pc.cpu().points_padded().numpy()[0])
    
    ## full point clouds
    full_pc_comparison = 0##pcu.one_sided_hausdorff_distance(a_full_pc[0].cpu().numpy(), b_full_pc.points_padded().cpu().numpy()[0])
    
    small_rmse = torch.sqrt(torch.mean(((small_a_pc - b_pc.points_padded())**2)))
    full_rmse = torch.sqrt(torch.mean(((a_full_pc[0] - b_full_pc.points_padded()[0])**2)))
    print(f"Reduced Hausdorff Reduced Data: {reduced_pc_comparison}")
    print(f"Directional Hausdorff Full Data: {full_pc_comparison}")
    print(f"Aligning points RMSE: {small_rmse}")
    print(f"Full pointcloud RMSE: {full_rmse}")
    
    
    # %%
    
    # Take a max of the one sided squared  distances to get the two sided Hausdorff distance
    general_hausdorff = 0##pcu.hausdorff_distance(a_full_pc[0].cpu().numpy(), b_full_pc.points_padded().cpu().numpy()[0])
    
    # # Find the index pairs of the two points with maximum shortest distancce
    # hausdorff_b_to_a, idx_b, idx_a = pcu.one_sided_hausdorff_distance(b, transformed_a, return_index=True)
    # assert np.abs(np.sum((a[idx_a] - b[idx_b])**2) - hausdorff_b_to_a**2) < 1e-5, "These values should be almost equal"
    # print(f"Hausdorff shortest: {hausdorff_a_to_b}")
    
    
    # # Find the index pairs of the two points with maximum shortest distancce
    # hausdorff_dist, idx_b, idx_a = pcu.hausdorff_distance(b, transformed_a, return_index=True)
    # assert np.abs(np.sum((a[idx_a] - b[idx_b])**2) - hausdorff_dist**2) < 1e-5, "These values should be almost equal"  
    # print(f"Hausdorff max: {hausdorff_a_to_b}")
    
    ##Save rotated point cloud
    print(f"Saving Mesh to {file_a[:-4]+f'matched{num_points}.obj'}")
    pcu.save_mesh_v((file_a[:-4]+f"matched{num_points}.obj"), a_full_pc[0].cpu().numpy())

    ##rmse.cpu().numpy()
    print("######################MATCHING COMPLETE########################")
    return full_rmse, reduced_pc_comparison, full_pc_comparison, general_hausdorff, time_taken, 0#iterations_taken
#%%
##Execute Testing of Hausdorff distanc function
results = pandas.DataFrame([],columns=['num_points','rmse','ICP_hausdorff','Full_1D_hausdorff','Full_2D_hausdorff','time_taken','iterations'])

test_obj = ["./data/nerf-synthetic/lego/lego_pointcloud_rotated.obj","./data/tandt/Ignatius/Ignatius_z_rot.obj","./data/tandt/Caterpillar/Caterpillar_shifted.obj","./data/tandt/chair/chair_RotatedPC.obj"]
GT_obj  = ["./data/nerf-synthetic/lego/lego_pointcloud.obj","./data/tandt/Ignatius/ignatius_base.obj","./data/tandt/Caterpillar/Caterpillar_base.obj","./data/tandt/chair/chair_BasePC.obj"]
for test,GT in zip(test_obj,GT_obj):
    for num_points in [100, 200, 1000,5000,7500]:
        rmse, reduced_pc_comparison, full_pc_comparison, general_hausdorff, time_taken, iterations_taken = IterationTest(test, GT, num_points)
        ##IterationTest( "./data/nerf-synthetic/lego/lego_pointcloud_rotated.obj", "./data/nerf-synthetic/lego/lego_pointcloud.obj", num_points)
        ##IterationTest("./data/tandt/Ignatius/Ignatius_z_rot.obj","./data/tandt/Ignatius/ignatius_base.obj",  num_points) ##IterationTest( "./data/tandt/Caterpillar/Caterpillar_shifted.obj","./data/tandt/Caterpillar/Caterpillar_base.obj", num_points)
        ##IterationTest("./data/tandt/chair/chair_BasePC.obj","./data/tandt/chair/chair_RotatedPC.obj",  num_points)
        
        ##IterationTest( "./data/nerf-synthetic/lego/lego_pointcloud_shrank_rotated.obj", "./data/nerf-synthetic/lego/lego_pointcloud_shrank.obj", num_points)
        new_row = {'Name':test_obj, 'num_points':num_points,'rmse':rmse, 'ICP_hausdorff':reduced_pc_comparison, 'Full_1D_hausdorff':full_pc_comparison, 'Full_2D_hausdorff':general_hausdorff, 'time_taken': time_taken, 'iterations': iterations_taken}
        
        results = results.append(new_row,ignore_index=True)
    print(results.to_string())
        
