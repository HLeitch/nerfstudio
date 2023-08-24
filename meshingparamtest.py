
import os
import sys

import numpy as np
import torch

if __name__ == "__main__":
    configDir = "C:\\Users\\hleit\\Documents\\nerfstudio\\outputs\\test-sphere\\nerfacto\\2023-04-27_164800/config.yml"
    loss1 = [0.0001]
    loss2 = (0.0001,0.00001,0.000001)
    loss3 = (0.0001,0.00001,0.000001)
    loss4 = (0.0001,0.00001,0.000001)

    for l1 in loss1:
        for l2 in loss2:
            for l3 in loss3:
                for l4 in loss4:
                    arr = f"{l1} {l2} {l3} {l4}"
                    os.system(
                        f"ns-export Marching-tet --load-config {configDir} --output-dir exports/spherelosstest/{l1}_{l2}_{l3}_{l4}/ --use-bounding-box True --bounding-box-min -0.15 -0.2 -0.35 --bounding-box-max 0.15 0.2 0.35 --loss_weights {l1} {l2} {l3} {l4}"
                    )

