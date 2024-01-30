import torch
from torch import nn
import os
import zipfile
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
from PIL import Image
import matplotlib.pyplot as plt

from utils.util import plot_transformed_images
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
    
train_dir = image_path / "train"
test_dir = image_path / "test"

random.seed(42) # <- try changing this and see what happens

# 1. Get all image paths (* means "any combination")
image_path_list = list(image_path.glob("*/*/*.jpg"))

data_transform = transforms.Compose([
  # Resize the images to 64x64
  transforms.Resize(size=(64, 64)),
  # Flip the images randomly on the horizontal
  transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
  # Turn the image into a torch.Tensor
  transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])

train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform)

class_names = train_data.classes
class_dict = train_data.class_to_idx

train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=1, # how many samples per batch?
                              num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=1, 
                             num_workers=1, 
                             shuffle=False) # don't usually need to shuffle testing data




# MODEL TIME

class TinyVGG(nn.Module):
  """ This is a direct COPY of the Tiny VGG model architecture!

  Args:
      nn (nn.Module): put in your hyperparameters here.
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
    super().__init__()
    self.conv_block_1 = nn.Sequential(
      nn.Conv2d(in_channels=input_shape, 
                out_channels=hidden_units, 
                kernel_size=3, # how big is the square that's going over the image?
                stride=1, # default
                padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,
                   stride=2)
    )
    self.conv_block_2 = nn.Sequential(
      nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features=hidden_units,
                out_features=output_shape)
    )
  def forward(self, x: torch.Tensor):
    x = self.conv_block_1(x)
    print(x.shape)
    x = self.conv_block_2(x)
    print(x.shape)
    x = self.classifier(x)
    print(x.shape)
    return x

torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3, hidden_units= 10, output_shape=len(train_data.classes)).to(device)
    