import os
import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged, ToTensord
from monai.data import CacheDataset, DataLoader
from glob import glob

def get_data_loader(data_dir, batch_size=2, mode="train"):
    # Define file paths
    images = sorted(glob(os.path.join(data_dir, mode, "images", "*.nii.gz")))
    labels = sorted(glob(os.path.join(data_dir, mode, "labels", "*.nii.gz")))

    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["image", "label"]),
    ])

    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=0.8, num_workers=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=="train"), num_workers=4)

    return dataloader
