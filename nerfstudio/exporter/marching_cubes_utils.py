import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import pywavefront as pwf
import torch
import tyro
from rich.console import Console
from typing_extensions import Annotated, Literal

CONSOLE = Console(width=120)


def save_obj(verts, normals, faces, output_dir, file_name):

    CONSOLE.print(f"[yellow]Saving mesh")

    file = open(output_dir.__str__() + "\\" + file_name, "w")
    for item in verts:
        file.write(f"v {item[0]} {item[1]} {item[2]}\n")
    for item in normals:
        file.write(f"vn {item[0]} {item[1]} {item[2]}\n")
    for item in faces:
        file.write(f"f {item[0]} {item[1]} {item[2]}\n")

    file.close()
    CONSOLE.print(f"[green]Mesh Written to {output_dir.__str__()}\\{file_name}")


##def refine_mesh():
