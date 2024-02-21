from __future__ import annotations

import os as os
import time as time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import GPUtil
import matplotlib.pyplot as plt
import mcubes as mcubes
import numpy as np
import pymeshlab
import skimage.io as skio
import tinycudann as tcnn
import torch as torch
import torch.nn.functional as F
import torch.utils.data as torch_data
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rich.console import Console
from skimage import measure as skmeasure
from torch.utils.data import DataLoader, SubsetRandomSampler, dataset
from torch.utils.tensorboard import SummaryWriter
from torchtyping import TensorType

import nerfstudio.exporter.marching_cubes_utils as mcUtils
import nerfstudio.fields.nerfacto_field
from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.exporter.exporter_utils import (
    Mesh,
    render_trajectory,
    render_trajectory_tri_tsdf,
)
from nerfstudio.exporter.object_renderer import render_mesh_to_tbfrom_components
from nerfstudio.exporter.ssan_dataset import SSANDataset
from nerfstudio.exporter.ssan_utils import SSANDataset, TSDFfromSSAN
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.nerfacto import NerfactoModelTriDepth
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.math import safe_normalize


###Due to compatiblity issues, a method seperate from tensorboard is needed to display density histograms.
###This method uses matplotlib and should save the output to a file in the same directory as the tensorboard data
###Returns: histogram numpy object
def display_histogram_of_densities(densities,file_destination,title="Density"):
    a = densities.reshape((1,-1))
    hist = plt.hist(a[0], bins=300)  # arguments are passed to np.histogram
    #plt.show()
    plt.title(f"{title} \n Avg: {np.average(a)}")
    plt.ylabel("Count")
    plt.xlabel("Density")
    plt.savefig(f"{file_destination}_{title}.png",dpi=400)

    plt.ylim(top=300)
    plt.savefig(f"{file_destination}_{title}_Shrunk.png",dpi=400)
    plt.clf()

    quantiled = np.array(a)
    np.quantile(a,0.9, out=quantiled)
    plt.hist(quantiled,bins=3000)
    plt.title(f"{title} 90th percentile \n Avg: {np.average(quantiled)}")

    plt.ylabel("Count")
    plt.xlabel("Density")
    #plt.xlim(left=0, right=10000)
    plt.savefig(f"{file_destination}_{title}_zoomed.png",dpi=400)
    return hist
        
