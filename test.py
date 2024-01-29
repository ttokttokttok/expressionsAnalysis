import torch
from torch import nn
import os
import zipfile
from pathlib import Path
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device type: {device}")

data_path = Path("data/")
image_path = data_path / "expressions"

if image_path.is_dir():
  print(f"{image_path} directory exists!")
else: 
  print(f"Did not find {image_path} creating one...")
  image_path.mkdir(parents=True, exist_ok=True)
  
  os.system(f"kaggle datasets download -d msambare/fer2013 -p {data_path}")
  with zipfile.ZipFile(data_path / "fer2013.zip", "r") as zip_ref:
    print("Unzipping expressions data") 
    zip_ref.extractall(image_path)
    
import os 
def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents. 

  Args:
      dir_path (str or pathlib.Path): target directory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    
walk_through_dir(image_path)