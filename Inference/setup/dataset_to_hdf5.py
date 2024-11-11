import os
import h5py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import numpy as np

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
dataset_path = os.path.join(os.getcwd(), '../Data/ImageNet/val')
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
batch_size = 500
data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=64, shuffle=False, pin_memory=True)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with h5py.File(os.path.join(dataset_path,'../processed_images.h5'), 'w') as h5f:
    # Set chunk size and compression for faster access
    chunk_size = (100, 3, 224, 224)
    images_ds = h5f.create_dataset(
        'images', 
        (len(dataset), 3, 224, 224), 
        dtype='float32',
        chunks=chunk_size
    )
    labels_ds = h5f.create_dataset('labels', (len(dataset),), dtype='int')

    # Use GPU for preprocessing and batch processing
    for i, (images, labels) in enumerate(data_loader):
        # Move data to GPU
        images, labels = images.to(device), labels.to(device)
        
        # Transfer to CPU before storing in HDF5 (necessary as h5py works only on CPU memory)
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()
        
        # Store images and labels in batches
        start_idx = i * batch_size
        end_idx = start_idx + images.shape[0]
        images_ds[start_idx:end_idx] = images
        labels_ds[start_idx:end_idx] = labels

        torch.cuda.empty_cache()

print("Data stored in HDF5 format.")
