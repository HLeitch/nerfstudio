
import os as os
import point_cloud_utils as pcu
import numpy as np
import torch as torch
import matplotlib.pyplot as plt

# Generate two random point sets
a = pcu.load_mesh_v("./data/tandt/Ignatius/ignatius_x_rot.obj")
b = pcu.load_mesh_v("./data/tandt/Ignatius/Ignatius_z_rot.obj")

fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")

ax.scatter(a[:,0],a[:,1],a[:,2],marker=".")
#ax.scatter(b[0],b[1],b[2],marker="o")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


plt.draw()
plt.close()


# %%
# Moving both sets of points to the origin.
avg_a = np.mean(a,0)
avg_b = np.mean(b,0)

print(f"{avg_a}, {avg_b}")
a -= avg_a
b -= avg_b

print(f"{np.mean(a,0)}, {np.mean(b,0)}")



# Compute one-sided squared Hausdorff distances
hausdorff_a_to_b = pcu.one_sided_hausdorff_distance(a, b)
hausdorff_b_to_a = pcu.one_sided_hausdorff_distance(b, a)

print(f"Hausdorff A to B: {hausdorff_a_to_b}")
print(f"Hausdorff B to A: {hausdorff_b_to_a}")


# %%

# Take a max of the one sided squared  distances to get the two sided Hausdorff distance
hausdorff_dist = pcu.hausdorff_distance(a, b)

# Find the index pairs of the two points with maximum shortest distancce
hausdorff_b_to_a, idx_b, idx_a = pcu.one_sided_hausdorff_distance(b, a, return_index=True)
assert np.abs(np.sum((a[idx_a] - b[idx_b])**2) - hausdorff_b_to_a**2) < 1e-5, "These values should be almost equal"
print(f"Hausdorff shortest: {hausdorff_a_to_b}")


# Find the index pairs of the two points with maximum shortest distancce
hausdorff_dist, idx_b, idx_a = pcu.hausdorff_distance(b, a, return_index=True)
assert np.abs(np.sum((a[idx_a] - b[idx_b])**2) - hausdorff_dist**2) < 1e-5, "These values should be almost equal"  
print(f"Hausdorff max: {hausdorff_a_to_b}")
