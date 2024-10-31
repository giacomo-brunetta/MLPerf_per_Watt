import torch
import time
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification
from datasets import load_dataset
import nvtx

# Dummy class to simulate ImageDataset behavior for clarity
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, processor):
        self.images = images
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # This is where you would preprocess the image using the processor
        return torch.rand(3, 224, 224)  # Dummy tensor for an image


# Main model inference function with profiling
def run_inference(batch_size=128, num_workers=4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)
    with nvtx.annotate("Data Loading:"):
        # Load the dataset
        dataset = load_dataset("yashikota/birds-525-species-image-classification")
        images = dataset["train"]["image"]
        print("With ", len(images), " images")

    # Load the processor and model, then move the model to the device (GPU or CPU)
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(device)

    # Skip DataParallel during profiling
    if torch.cuda.device_count() > 1:
        print("Skipping DataParallel during profiling for simplicity.")

    # Create a DataLoader with custom ImageDataset and preprocessing
    image_dataset = ImageDataset(images, processor)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True  # Enable pinned memory for faster transfers
    )

    predicted_labels = []

    start_time = time.perf_counter()
    with nvtx.annotate("Training:"):
        # Process images in batches from the DataLoader
        for batch_images in dataloader:
            # Transfer the batch to the GPU asynchronously
            batch_images = batch_images.to(device, non_blocking=True)

            # Perform inference
            with torch.no_grad():
                logits = model(batch_images).logits

            predicted_batch_labels = logits.argmax(-1).tolist()
            predicted_labels.extend([model.config.id2label[label] for label in predicted_batch_labels])

    end_time = time.perf_counter()

    inference_time = end_time - start_time
    print("Inference time:", inference_time, "s")
    print("Throughput: ", len(images) / inference_time, " imgs/s")


if __name__ == "__main__":
    run_inference(batch_size=512, num_workers=25)
