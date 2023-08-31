
import os
import sys

import numpy as np
import torch



def loss_test():
    configDir = "C:\\Users\\hleit\\Documents\\nerfstudio\\outputs\\data\\TandT\\Ignatius\\nerfacto\\2023-08-15_135545\\config.yml"
    loss1 = [0.01,0.001]
    loss2 = (0.001,0.0001,0.00001)
    loss3 = (0.001,0.0001,0.00001)
    loss4 = (0.001,0.0001,0.00001)

    for l1 in loss1:
        for l2 in loss2:
            for l3 in loss3:
                for l4 in loss4:
                    arr = f"{l1} {l2} {l3} {l4}"
                    os.system(
                        f"ns-export Marching-tet --load-config {configDir} --output-dir exports/sphere_loss_test/{l1}_{l2}_{l3}_{l4}/ --use-bounding-box True --bounding-box-min -0.004000000000000031 -0.25200000000000006 -0.11099999999999997 --bounding-box-max 0.30400000000000005 0.04799999999999993 0.189 --epochs 12 --batch_splits 50 --loss_weights {l1} {l2} {l3} {l4} --nerf_image_path=ssan/testsphere"
                    )

def epoch_division_Test_sphere():
    configDir = "C:\\Users\\hleit\\Documents\\nerfstudio\\outputs\\test-sphere\\nerfacto\\2023-04-27_164800/config.yml"
    epochs = [10,12,15,20,25]
    divisions = [20,50,100,150]
    for e in epochs:
        for d in divisions:
            os.system(
                        f"ns-export Marching-tet --load-config {configDir} --output-dir exports/sphere_epoch_test/Epoch{e}_Div{d}/ --use-bounding-box True --bounding-box-min -0.004000000000000031 -0.25200000000000006 -0.11099999999999997 --bounding-box-max 0.30400000000000005 0.04799999999999993 0.189 --epochs {e} --batch_splits {d} --loss_weights 0.01 0.00001 0.00001 0.00001 --nerf_image_path=ssan/testsphere"
                    )
def epoch_division_Test_statue():
    configDir = "C:\\Users\\hleit\\Documents\\nerfstudio\\outputs\\data\\TandT\\Ignatius\\nerfacto\\2023-08-15_135545\\config.yml"
    epochs = [10,12,15,20,25]
    divisions = [20,50,100,150]
    for e in epochs:
        for d in divisions:
            os.system(
                        f"ns-export Marching-tet --load-config {configDir} --output-dir exports/statue_hires_epoch_test/Epoch{e}_Div{d}/ --use-bounding-box True --bounding-box-min -0.15 -0.2 -0.35 --bounding-box-max 0.15 0.2 0.35 --epochs {e} --batch_splits {d} --loss_weights 0.01 0.001 0.001 0.001 --nerf_image_path=ssan/teststatue"
                    )

if __name__ == "__main__":
    loss_test()