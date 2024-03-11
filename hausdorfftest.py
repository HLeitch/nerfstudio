
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
from torch.utils.tensorboard import SummaryWriter


def IterationTest(file_a, file_b, num_points, comparison_points, tb_file: SummaryWriter):

    tb_file.add_text("Test File",file_a)
    tb_file.add_text("GT File", file_b)
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
    
    ##_full added to subject to explore how changing num of points changes results
    a_tensor_full = a_tensor[:comparison_points]#[None,:,:]
    b_tensor_full = b_tensor[:comparison_points]#[None,:,:]

    ##just for testing correspondance
    # pointsA = torchops.sample_farthest_points(a_tensor,K=num_points)[0]
    # pointsB = torchops.sample_farthest_points(b_tensor, K=num_points)[0]

    ##sampling comparison scatter plot

    fig = plt.figure()  
    ax = fig.add_subplot(111,projection="3d")
    fig.suptitle("Pre-Alignment Point Cloud Comparison")
    
    furthest_points = ax.scatter(pointsA[0,:,0]+0.05,pointsA[0,:,1],pointsA[0,:,2],marker=".",label="Test Points",s=0.001)
    random_points = ax.scatter(pointsB[0,:,0],pointsB[0,:,1],pointsB[0,:,2],marker="s",color="r",label="GT Points",s=0.001)
    ax.legend(handles=[furthest_points,random_points])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    tb_file.add_figure("Unaligned Cloud",fig,num_points)
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

    iterations_taken = output.t_history.__len__()
    tb_file.add_scalar("Iterations Taken",iterations_taken,num_points)

    ##print(f"iterations_taken {iterations_taken}")
    
    ## reduced a points only
    small_a_pc = RTs.s[:,None,None] * torch.bmm(a_pc.points_padded(), RTs.R) + RTs.T[:,None,:]
    
    fig = plt.figure()  
    ax = fig.add_subplot(111,projection="3d")
    fig.suptitle("Aligned Point Cloud Comparison")
    
    furthest_points = ax.scatter(small_a_pc[0,:,0],small_a_pc[0,:,1],small_a_pc[0,:,2],marker=".",label="Rotated Points")
    random_points = ax.scatter(pointsB[0,:,0],pointsB[0,:,1],pointsB[0,:,2],marker="s",color="r",label="Base Points")
    ax.legend(handles=[furthest_points,random_points])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    tb_file.add_figure("Aligned Cloud",fig,num_points)


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
    
    general_hausdorff = 0#pcu.hausdorff_distance(test_PCD[0].cpu().numpy(), gt_PCD.points_padded().cpu().numpy()[0]

    small_rmse = torch.sqrt(torch.mean(((small_a_pc - b_pc.points_padded())**2)))
    full_rmse = torch.sqrt(torch.mean(((a_full_pc[0] - b_full_pc.points_padded()[0])**2)))
    print(f"Reduced Hausdorff Reduced Data: {reduced_pc_comparison}")
    print(f"Directional Hausdorff Full Data: {full_pc_comparison}")
    print(f"Aligning points RMSE: {small_rmse}")
    print(f"Full pointcloud RMSE: {full_rmse}")
    
    
    # %%
    ##Save rotated point cloud
    print(f"Saving Mesh to {file_a[:-4]+f'matched{num_points}.obj'}")
    ##pcu.save_mesh_v((file_a[:-4]+f"matched{num_points}.obj"), a_full_pc[0].cpu().numpy())
    PCD_Comparison(a_full_pc,b_full_pc,tb_file=tb_file)

    ##rmse.cpu().numpy()
    print("######################MATCHING COMPLETE########################")
    time_taken = time.time() - time_start
    tb_file.add_scalar("Time Taken",time_taken,num_points)
    return full_rmse, reduced_pc_comparison, full_pc_comparison, general_hausdorff, time_taken, iterations_taken
#%%
def PCD_Comparison(test_PCD, gt_PCD, tb_file: SummaryWriter):
    general_hausdorff = pcu.hausdorff_distance(test_PCD[0].cpu().numpy(), gt_PCD.points_padded().cpu().numpy()[0])
    tb_file.add_scalar("General Hausdorff",general_hausdorff,num_points)
    
    # Find the index pairs of the two points with maximum shortest distancce
    hausdorff_b_to_a, idx_b, idx_a = pcu.one_sided_hausdorff_distance(test_PCD[0].cpu().numpy(), gt_PCD.points_padded().cpu().numpy()[0], return_index=True)
    #assert np.abs(np.sum((a[idx_a] - b[idx_b])**2) - hausdorff_b_to_a**2) < 1e-5, "These values should be almost equal"
    print(f"Hausdorff shortest: {hausdorff_b_to_a}")
    tb_file.add_scalar("Shortest from GT to Test",hausdorff_b_to_a,num_points)
    
    
    # Find the index pairs of the two points with maximum shortest distancce
    hausdorff_dist, idx_b, idx_a = pcu.hausdorff_distance(test_PCD[0].cpu().numpy(), gt_PCD.points_padded().cpu().numpy()[0], return_index=True)
    #assert np.abs(np.sum((a[idx_a] - b[idx_b])**2) - hausdorff_dist**2) < 1e-5, "These values should be almost equal"  
    print(f"Hausdorff max: {hausdorff_dist}")
    tb_file.add_scalar("Maximum Shortest Distance",hausdorff_dist,num_points)


    ## root
    full_rmse = torch.sqrt(torch.mean(((test_PCD[0] - gt_PCD.points_padded()[0])**2)))
    tb_file.add_scalar("RMSE value",full_rmse,num_points)

    chamfer_dist_output = pcu.chamfer_distance(test_PCD[0].cpu().numpy(), gt_PCD.points_padded().cpu().numpy()[0])
    tb_file.add_scalar("Chamfer Distance",chamfer_dist_output,num_points)
    
##Execute Testing of Hausdorff distanc function
results = pandas.DataFrame([],columns=['num_points','rmse','ICP_hausdorff','Full_1D_hausdorff','Full_2D_hausdorff','time_taken','iterations'])

test_obj = ["./data/nerf-synthetic/lego/lego_pointcloud_rotated.obj","./data/tandt/Ignatius/Ignatius_z_rot.obj","./data/tandt/Caterpillar/Caterpillar_shifted.obj","./data/nerf-synthetic/chair/chair_RotatedPC.obj"]
GT_obj  = ["./data/nerf-synthetic/lego/lego_pointcloud.obj","./data/tandt/Ignatius/ignatius_base.obj","./data/tandt/Caterpillar/Caterpillar_base.obj","./data/nerf-synthetic/chair/chair_BasePC.obj"]

# test_obj = ["./data/nerf-synthetic/chair/chair_RotatedPC.obj"]
# GT_obj = ["./data/nerf-synthetic/chair/chair_BasePC.obj"]

for test,GT in zip(test_obj,GT_obj):
    for num_points in [200, 1000,5000]:
        for comp_points in [5000,10000,25000,50000,100000,200000,500000]:
            tb_file  = SummaryWriter(f"Mesh_Analysis_Test/Comp_points/{test.split(sep='/')[-1]}_{comp_points}")
            rmse, reduced_pc_comparison, full_pc_comparison, general_hausdorff, time_taken, iterations_taken = IterationTest(test, GT, num_points,comp_points,tb_file=tb_file)
            ##IterationTest( "./data/nerf-synthetic/lego/lego_pointcloud_rotated.obj", "./data/nerf-synthetic/lego/lego_pointcloud.obj", num_points)
            ##IterationTest("./data/tandt/Ignatius/Ignatius_z_rot.obj","./data/tandt/Ignatius/ignatius_base.obj",  num_points) ##IterationTest( "./data/tandt/Caterpillar/Caterpillar_shifted.obj","./data/tandt/Caterpillar/Caterpillar_base.obj", num_points)
            ##IterationTest("./data/tandt/chair/chair_BasePC.obj","./data/tandt/chair/chair_RotatedPC.obj",  num_points)
            
            ##IterationTest( "./data/nerf-synthetic/lego/lego_pointcloud_shrank_rotated.obj", "./data/nerf-synthetic/lego/lego_pointcloud_shrank.obj", num_points)
            new_row = {'Name':test, 'num_points':num_points,'rmse':rmse, 'ICP_hausdorff':reduced_pc_comparison, 'Full_1D_hausdorff':full_pc_comparison, 'Full_2D_hausdorff':general_hausdorff, 'time_taken': time_taken, 'iterations': iterations_taken}
            
            results = results.append(new_row,ignore_index=True)
    print(results.to_string())
print(results.to_string())

