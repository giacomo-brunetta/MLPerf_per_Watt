import os
import h5py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
data_loader = DataLoader(dataset, batch_size=1, num_workers=64, shuffle=False)

with h5py.File('processed_images.h5', 'w') as h5f:
    # Set chunk size and compression for faster access
    chunk_size = (500, 3, 224, 224)  # Adjust based on available memory
    images_ds = h5f.create_dataset(
        'images', 
        (len(dataset), 3, 224, 224), 
        dtype='float32',
        chunks=chunk_size,
        compression="gzip",  # Options: "gzip", "lzf", or "szip"
        compression_opts=9    # Higher values mean more compression (0-9 for gzip)
    )
    labels_ds = h5f.create_dataset('labels', (len(dataset),), dtype='int')
    
    # Store images and labels as before
    for i, (image, label) in enumerate(data_loader):
        images_ds[i] = image.numpy()
        labels_ds[i] = label.numpy()

print("Data stored in HDF5 format.")