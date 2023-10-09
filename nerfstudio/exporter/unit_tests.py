from nerfstudio.exporter.ssan_utils import SSANDataset
from nerfstudio.exporter.ssan_utils import TSDFfromSSAN

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
from nerfstudio.exporter.object_renderer import render_mesh
from nerfstudio.exporter.ssan_dataset import SSANDataset
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.nerfacto import NerfactoModelTriDepth
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.math import safe_normalize


def test(ssanNetwork: TSDFfromSSAN, dataset: SSANDataset,tensorboard: SummaryWriter):
    
