from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
import os
from utils.profiling_utils import Profiler, profile_with
import argparse

import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file_path):
        self.h5_file = h5py.File(hdf5_file_path, 'r')
        self.images = self.h5_file['images']
        self.labels = self.h5_file['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load image and label asynchronously
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

    def close(self):
        self.h5_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile ResNet50")

    # Adding arguments
    parser.add_argument("device", type=str, help="The device to run on (cpu | cuda)")
    parser.add_argument("profiler", type=str, help="The profiler (power | torch | nsight)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--precision", type=str, default='float', help= "fp32 | tf32 | fp16")
    parser.add_argument("--workers", type=int, default=25, help="Workers of dataloader")
    parser.add_argument("--compile", type=bool, default=True, help="Compile the model")
    parser.add_argument("--hdf5", type=bool, default=True, help="Use dataset in HDF5 format")

    # Parse arguments
    args = parser.parse_args()

    device = args.device
    profiler = Profiler.from_str(args.profiler)
    batch_size = args.batch_size
    workers = args.workers
    compile = args.compile
    precision = args.precision
    hadoop = args.hdf5

    print("Batch size: ", batch_size)
    print("Workers: ", workers)
    print("precision: ", precision)

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
    ])

    # get the dataset

    if hadoop:
        hdf5_file_path = os.path.join(os.getcwd(), '../Data/ImageNet/processed_images.h5')
        dataset = HDF5Dataset(hdf5_file_path)
    else: 
        dataset_path = os.path.join(os.getcwd(), '../Data')
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    if profiler:
        profile_with(profiler,
                     model,
                     dataset,
                     device,
                     batch_size=batch_size,
                     num_workers=workers,
                     interval=0.1,
                     compile=compile,
                     precision=precision)
    else:
        print("Select a valid profiler (power | torch | nsight)")
    
    if hadoop:
        dataset.close()
