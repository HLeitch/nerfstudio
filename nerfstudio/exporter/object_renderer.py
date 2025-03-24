import io

import numpy as np
import open3d
import open3d as o3d
import open3d.visualization as vis
import pymeshlab
import tensorboard as tb
import torch
from matplotlib import pyplot as plt
from PIL import Image


def render_mesh_to_tbfrom_components(verts ,triangles ,norms,filename:str, profiler):
    
    mesh = open3d.geometry.TriangleMesh()
    print(verts.shape)
    verts = open3d.utility.Vector3dVector(verts)
    triangles = open3d.utility.Vector3iVector(triangles)
    mesh = open3d.geometry.TriangleMesh(verts,triangles)
    mesh.compute_vertex_normals()
    v = vis.Visualizer()
    v.create_window(width=1920,height=1080)#window_name = "render",visable = True)
    v.add_geometry(mesh)
    v.capture_screen_image(f"{filename}.png",do_render=True)
    v.destroy_window()

    img = Image.open(f"{filename}.png")
    img = np.asarray(img).T
    profiler.add_image(f"MeshRender/{filename}",img)

def render_mesh_to_tb(mesh: open3d.geometry.TriangleMesh,filename:str,profiler):
    v = vis.Visualizer()
    v.create_window(width=1920,height=1080)#window_name = "render",visable = True)
    v.add_geometry(mesh)
    v.capture_screen_image(f"{filename}.png",do_render=True)
    v.destroy_window()

    img = Image.open(f"{filename}.png")
    img = np.asarray(img).T
    profiler.add_image(f"MeshRender/{filename}",img)

# if __name__ == "__main__":
    