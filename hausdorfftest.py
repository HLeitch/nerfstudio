
import os as os

import matplotlib.pyplot as plt
import numpy as np
import point_cloud_utils as pcu
import pytorch3d as torch3d
import pytorch3d.ops as torchops
from pytorch3d.structures import Pointclouds
import torch as torch

def IterationTest(file_a, file_b):
    
    # Generate two random point sets
    a = pcu.load_mesh_v(file_a)
    b = pcu.load_mesh_v(file_b)
    
    
    
    
    a_tensor_full = torch.tensor(a,dtype=torch.float)
    b_tensor_full = torch.tensor(b,dtype=torch.float)

    
    randomPoints = torch.randperm(a_tensor_full.shape[0])
    
    ##randomly select 4000 points 
    a_tensor = a_tensor_full[randomPoints]
    b_tensor = b_tensor_full[randomPoints]
    
    a_tensor = a_tensor[:10000][None,:,:]
    b_tensor = b_tensor[:10000][None,:,:]
    
    print(f"{a_tensor.shape}")
    #result = 
    
    # %%
    
    # Moving both sets of points to the origin.
    avg_a = np.mean(a_tensor.numpy(),0)
    avg_b = np.mean(b_tensor.numpy(),0)
    
    print(f"{avg_a}, {avg_b}")
    a = avg_a
    b = avg_b
    
    print(f"{np.mean(a,0)}, {np.mean(b,0)}")
    fig = plt.figure()  
    ax = fig.add_subplot(111,projection="3d")
    
    ax.scatter(a[:,0],a[:,1],a[:,2],marker=".")
    ##ax.scatter(b[:,0],b[:,1],b[:,2],marker=".")
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
    
    output = torchops.iterative_closest_point(a_pc, b_pc,verbose=True,max_iterations=200)
    rmse = output.rmse
    transformed_a = output.Xt
    RTs = output.RTs
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_alignment.html#iterative_closest_point
    a_full_pc
    #%%
    print(f"converted: {output.converged}, rmse{output.rmse}")
    print(f"simTransform {output.RTs}")
    
    #%%
    print(f"xT = {output.Xt}")
    ##transformed = output.Xt.points_list()
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection="3d")
    
    ax.scatter(transformed_a[0][:,0],transformed_a[0][:,1],transformed_a[0][:,2],marker=".")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    
    plt.show()
    
    # %%
    # Compute one-sided squared Hausdorff distances
    hausdorff_a_to_b = pcu.one_sided_hausdorff_distance(transformed_a, b)
    hausdorff_b_to_a = pcu.one_sided_hausdorff_distance(b, transform_a)
    
    print(f"Hausdorff A to B: {hausdorff_a_to_b}")
    print(f"Hausdorff B to A: {hausdorff_b_to_a}")
    
    
    # %%
    
    # Take a max of the one sided squared  distances to get the two sided Hausdorff distance
    hausdorff_dist = pcu.hausdorff_distance(transformed_a, b)
    
    # Find the index pairs of the two points with maximum shortest distancce
    hausdorff_b_to_a, idx_b, idx_a = pcu.one_sided_hausdorff_distance(b, transformed_a, return_index=True)
    assert np.abs(np.sum((a[idx_a] - b[idx_b])**2) - hausdorff_b_to_a**2) < 1e-5, "These values should be almost equal"
    print(f"Hausdorff shortest: {hausdorff_a_to_b}")
    
    
    # Find the index pairs of the two points with maximum shortest distancce
    hausdorff_dist, idx_b, idx_a = pcu.hausdorff_distance(b, transformed_a, return_index=True)
    assert np.abs(np.sum((a[idx_a] - b[idx_b])**2) - hausdorff_dist**2) < 1e-5, "These values should be almost equal"  
    print(f"Hausdorff max: {hausdorff_a_to_b}")
    
    return rmse, hausdorff_dist
#%%

u,v = IterationTest("./data/tandt/Ignatius/ignatius_x_rot.obj", "./data/tandt/Ignatius/Ignatius_z_rot.obj")
    
