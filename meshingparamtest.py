
import os
import sys

import numpy as np
import torch
import time



def loss_test():
    configDir = "C:\\Users\\hleit\\Documents\\nerfstudio\\outputs\\data\\TandT\\Ignatius\\nerfacto\\2023-08-15_135545\\config.yml"
    loss1 = [0.001]
    loss2 = (0.0001,0.00001)
    loss3 = (0.0001,0.00001)
    loss4 = [0.000]
    
    t = time.localtime()
    timestring = f"{t.tm_mday}-{t.tm_mon}_{t.tm_hour}-{t.tm_min}"

    for l1 in loss1:
        for l2 in loss2:
            for l3 in loss3:
                for l4 in loss4:
                    arr = f"{l1} {l2} {l3} {l4}"
                    os.system(
                        f"ns-export Marching-tet --load-config {configDir} --output-dir exports/statue_loss_{timestring}/{l1}_{l2}_{l3}_{l4}/ --use-bounding-box True --bounding-box-min -0.15 -0.2 -0.35 --bounding-box-max 0.15 0.2 0.35 --epochs 12 --batch_splits 50 --loss_weights {l1} {l2} {l3} {l4} --nerf_image_path=ssan/teststatue"
                    )

def epoch_division_Test_sphere():
    configDir = "C:\\Users\\hleit\\Documents\\nerfstudio\\outputs\\data\\Unreal_Sphere\\nerfacto\\2023-09-09_151855/config.yml"
    epochs = [18]
    divisions = [20,50,100,150]
    t = time.localtime()
    timestring = f"{t.tm_mday}-{t.tm_mon}_{t.tm_hour}-{t.tm_min}"

    for e in epochs:
        for d in divisions:
            os.system(
                        f"ns-export Marching-tet --load-config {configDir} --output-dir exports/unreal_sphere_epoch_test_{timestring}/Epoch{e}_Div{d}/ --use-bounding-box True --bounding-box-min -0.25 -0.09 -0.15 --bounding-box-max 0.05 0.2 0.15 --epochs {e} --batch_splits {d} --loss_weights 0.01 0.001 0.001 0.0 --nerf_image_path=ssan/Unrealtestsphere"
                    )
def epoch_division_Test_statue():
    configDir = "C:\\Users\\hleit\\Documents\\nerfstudio\\outputs\\data\\TandT\\Ignatius\\nerfacto\\2023-08-15_135545\\config.yml"
    epochs = [10,12,15,20,25]
    divisions = [20,50,100,150]
    t = time.localtime()
    timestring = f"{t.tm_mday}_{t.tm_mon}_{t.tm_year}_{t.tm_hour}:{t.tm_min}"
    for e in epochs:
        for d in divisions:
            os.system(
                        f"ns-export Marching-tet --load-config {configDir} --output-dir exports/statue_hires_epoch_test/Epoch{e}_Div{d}_{timestring}/ --use-bounding-box True --bounding-box-min -0.15 -0.2 -0.35 --bounding-box-max 0.15 0.2 0.35 --epochs {e} --batch_splits {d} --loss_weights 0.01 0.001 0.001 0.001 --nerf_image_path=ssan/teststatue"
                    )

if __name__ == "__main__":
    epoch_division_Test_sphere()