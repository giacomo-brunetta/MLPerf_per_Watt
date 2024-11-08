from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
import os
from utils.profiling_utils import Profiler, profile_with
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile ResNet50")

    # Adding arguments
    parser.add_argument("device", type=str, help="The device to run on (cpu | cuda)")
    parser.add_argument("profiler", type=str, help="The profiler (power | torch | nsight)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--workers", type=int, default=25, help="Workers of dataloader")

    # Parse arguments
    args = parser.parse_args()

    device = args.device
    profiler = Profiler.from_str(args.profiler)
    batch_size = args.batch_size
    workers = args.workers

    print("Batch size: ", batch_size)
    print("Workers: ", workers)

    model = models.detection.retinanet_resnet50_fpn(pretrained=True)
    
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
    ])

    # get the dataset
    dataset_path = os.path.join(os.getcwd(), '../Data/ImageNet/val')

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    if profiler:
        profile_with(profiler, model, dataset, device, batch_size=batch_size, num_workers=workers, interval=0.1)
    else:
        print("Select a valid profiler (power | torch | nsight)")