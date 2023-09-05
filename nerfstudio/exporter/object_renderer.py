import numpy as np
from matplotlib import pyplot as plt
import pymeshlab

import torch



def render_mesh(verts: torch.Tensor,faces: torch.Tensor,norms: torch.Tensor,path:str):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(verts,faces)
    ax.set_aspect('equal')
    plt.savefig(f"{path}/mesh_render.png")
    
    pymeshlab.

if __name__ == "__main__":
    