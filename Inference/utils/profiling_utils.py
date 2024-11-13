import time
import matplotlib.pyplot as plt
import torch
import numpy as np
from enum import Enum
from torch.utils.data import DataLoader

# profilers
from utils.power_utils import gpuPowerProbe
import torch.profiler as profiler
import nvtx

class Profiler(Enum):
    POWER = 0
    NSIGHT = 1
    TORCH = 2

    def from_str(string):
        if string.lower() == "power":
            return Profiler.POWER
        elif string.lower() == "nsight":
            return Profiler.NSIGHT
        elif string.lower() == "torch":
            return Profiler.TORCH
        else:
            return None

def setup(model, dataset, device_name, batch_size, num_workers, compile, precision):
    # Check if GPUs are available, otherwise fallback to CPU
    device = torch.device(device_name)
    print("Running on: ", device)

    torch.backends.cudnn.enabled = True

    if precision == 'fp32':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif precision == 'tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    # Load the model, then move the model to the device (GPU or CPU)
    model.eval()
    model.to(device)

    if precision == 'fp16':
        model = model.half()

    # Wrap the model for DataParallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)  # Use DataParallel for multi-GPU
    
    # If possible compile
    if compile:
        print("With compiled model")
        model = torch.compile(model)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return device, model, dataloader

def inference(model, device, dataloader, precision, profiler = None):
    correct = 0
    batch_loaded = []

    start = time.perf_counter()

    # Process images in batches from the DataLoader
    for batch_images, batch_labels in dataloader:  # Unpack images and labels
        # Transfer the batch to the GPU asynchronously
        batch_images = batch_images.to(device, non_blocking=True)
        if precision == 'fp16':
            batch_images = batch_images.half()
        batch_labels = batch_labels.to(device, non_blocking=True)

        end = time.perf_counter()
        batch_loaded.append(end-start)

        # Perform inference
        with torch.no_grad():
            logits = model(batch_images)
        
        # Predict labels for the batch
        predicted_batch_labels = logits.argmax(dim=-1)

        # Count correct predictions
        correct += (predicted_batch_labels == batch_labels).sum().item()
        if profiler != None:
            profiler.step()  # Advance profiler's steps after each batch
    
    return correct, np.array(batch_loaded)

def power_profile_inference(model, dataset, device_name, interval, batch_size, num_workers, compile, precision):

    device, model, dataloader = setup(model, dataset, device_name, batch_size, num_workers, compile, precision)

    # Profile with gpuPowerProbe
    power_profiles = []
    for id in range(torch.cuda.device_count()):
        power_profiles.append(gpuPowerProbe(interval=interval, gpu_id=id))
        power_profiles[id].start()

    start_time = time.perf_counter()
    # Actual workload
    correct, batch_loaded = inference(model, device, dataloader, precision)

    latency = time.perf_counter() - start_time

    inference_powers = []
    inference_powers_time = []

    #stop power profile
    for power_profile in power_profiles:
        power, times = power_profile.stop()
        inference_powers.append(power)
        inference_powers_time.append(times)
        power_profile.destroy()
    
    # Stats
    total_imgs = len(dataset)
    throughput =  total_imgs/latency
    accuracy = correct/total_imgs

    # Print Stats
    print("---------------------------------------------")
    print("Results")
    print("---------------------------------------------")
    print(f"Tot imgs: {total_imgs}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Latency: {latency :.2f} s")
    print(f"Throughput: {throughput :.2f} imgs/s")

    avg_total_power = 0
    
    plt.figure(figsize=(100,5))

    for id in range(torch.cuda.device_count()):
        print(f"GPU {id}:")
        power = np.array(inference_powers[id]) / 1000 # to Watt
        times = np.array(inference_powers_time[id])
        avg_power = np.mean(power)
        avg_total_power += avg_power
        peak_power = np.max(power)
        energy = np.sum(power*times)

        print(f"    Power avg: {avg_power :.2f} W")
        print(f"    Power peak: {peak_power :.2f} W")
        print(f"    Energy: {energy :.2f} J")

        # Plot power consumption over time
        plt.plot(np.cumsum(inference_powers_time[id]), power, label=f'GPU {id}')

    print(f"Imgs/J: {float(throughput/avg_total_power) :.2f}")

    for x in batch_loaded:
        plt.axvline(x=x, color='b', linestyle=':', linewidth=1) 

    plt.legend()
    plt.xlabel(f'Time ({interval} sec intervals)')
    plt.ylabel('Power Consumption (W)')
    plt.savefig('gpu_power_plot.png')

def torch_profile_inference(model, dataset, device_name, batch_size, num_workers, compile, precision):
    device, model, dataloader = setup(model, dataset, device_name, batch_size, num_workers, compile, precision)

    start_time = time.perf_counter()
    # Using torch profiler

    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        schedule=profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=profiler.tensorboard_trace_handler("./log"),
        record_shapes=True,
        profile_memory=True,
        with_stack=False  # Disable stack tracing to avoid the replay stack issue
    ) as prof:
        correct, _ = inference(model, device, dataloader, precision, profiler=prof)
        pass

    # Stats
    latency = time.perf_counter()-start_time
    total_imgs = len(dataset)
    throughput =  total_imgs/latency
    accuracy = correct/total_imgs

    # Print Stats
    print("---------------------------------------------")
    print("Results")
    print("---------------------------------------------")
    print(f"Tot imgs: {total_imgs}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Latency: {latency :.2f} s")
    print(f"Throughput: {throughput :.2f} imgs/s")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

def nsight_profile_inference(model, dataset, device_name, batch_size, num_workers, compile, precision):
    device, model, dataloader = setup(model, dataset, device_name, batch_size, num_workers, compile, precision)

    # Actual workload
    start_time = time.perf_counter()
    correct = 0

    # Process images in batches from the DataLoader
    with nvtx.annotate("Inference:"):
        correct, _ = inference(model, device, dataloader, precision)

    latency = time.perf_counter() - start_time
    
    # Stats
    total_imgs = len(dataset)
    throughput =  total_imgs/latency
    accuracy = correct/total_imgs

    # Print Stats
    print("---------------------------------------------")
    print("Results")
    print("---------------------------------------------")
    print(f"Tot imgs: {total_imgs}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Latency: {latency :.2f} s")
    print(f"Throughput: {throughput :.2f} imgs/s")

def profile_with(profiler, model, dataset, device_name, interval, batch_size, num_workers, compile, precision):
    if profiler == Profiler.POWER:
        power_profile_inference(model, dataset, device_name, interval, batch_size, num_workers, compile, precision)
    elif profiler == Profiler.TORCH:
        torch_profile_inference(model, dataset, device_name, batch_size, num_workers, compile, precision)
    elif profiler == Profiler.NSIGHT:
        nsight_profile_inference(model, dataset, device_name, batch_size, num_workers, compile, precision)
    else:
        print("Please select a valid profiler")