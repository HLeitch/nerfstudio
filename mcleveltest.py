# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 09:49:43 2023

@author: hleit
"""

import numpy as np
import torch
import os
import sys



if __name__ == "__main__":
    configDir= "C:\\Users\\hleit\\Documents\\nerfstudio\\outputs\\data\\TandT\\ignatius\\nerfacto\\2023-02-16_131852\\config.yml"
    for i in {0,1,2,3,4,6,8,10,15,20,25,30,40,50,60,70,80,90,100}:
        os.system(f"ns-export marching-cubes --load-config {configDir} --output-dir exports/mesh/ --use-bounding-box True --bounding-box-min -0.27 -0.25 -0.25 --bounding-box-max 0.33 0.36 0.8 --num-samples=256 --save_mesh=True --output-file-name level{i}.obj  --mc_level={i}")