# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 09:49:43 2023

@author: hleit
"""

import os
import sys

import numpy as np
import torch

if __name__ == "__main__":
    configDir = "C:\\Users\\hleit\\Documents\\nerfstudio\\outputs\\data\\TandT\\ignatius\\instant-ngp-bounded\\2023-03-06_125519\\config.yml"
    for i in {3, 6, 9, 12, 15, 20}:
        os.system(
            f"ns-export marching-cubes --load-config {configDir} --output-dir exports/mesh/ --use-bounding-box True --bounding-box-min -0.15 -0.25 -0.75 --bounding-box-max 0.35 0.25 0.75 --num-samples=256 --save_mesh=True --output-file-name level{i}.obj  --mc_level={i}"
        )
