import numpy as np

##import open3d as o3d
import open3d
import open3d.visualization as vis
import pymeshlab
import torch
from matplotlib import pyplot as plt


def render_mesh(verts ,triangles ,norms,path:str):
    
    mesh = open3d.geometry.TriangleMesh()
    print(verts.shape)
    verts = open3d.utility.Vector3dVector(verts)
    triangles = open3d.utility.Vector3iVector(triangles)
    mesh = open3d.geometry.TriangleMesh(verts,triangles)
    mesh.compute_vertex_normals()
    v = vis.Visualizer()
    v.create_window()#window_name = "render",visable = True)
    v.add_geometry(mesh)
    img = v.capture_screen_float_buffer(do_render=False)

    open3d.io.write_image(f"render.png",img,-1)
    ##plt.imshow(np.asanyarray(img))

    open3d.visualization.draw(mesh,f"{path}",1920,1080)
# if __name__ == "__main__":
    