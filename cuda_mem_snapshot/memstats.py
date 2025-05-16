import torch
import os
import matplotlib.pyplot as plt

from datetime import datetime

MEMSTATS_OUPUT_FILE = os.getenv("MEMSTATS_OUPUT_FILE", "memstats.txt")

def capture_mem_stats(device, label):
    with open(MEMSTATS_OUPUT_FILE, 'a') as f:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{device}::{ts}::{label}::allocator_backend: {torch.cuda.get_allocator_backend()}")
        f.write(f"{device}::{ts}::{label}::device_memory_used:  {torch.cuda.device_memory_used(device) / (1024**3)}")
        f.write(f"{device}::{ts}::{label}::memory_allocated:  {torch.cuda.memory_allocated(device) / (1024**3)}")
        f.write(f"{device}::{ts}::{label}::memory_reserved:  {torch.cuda.memory_reserved(device) / (1024**3)}")
        f.write(f"{device}::{ts}::{label}::max_memory_allocated:  {torch.cuda.max_memory_allocated(device) / (1024**3)}")
        f.write(f"{device}::{ts}::{label}::max_memory_reserved:  {torch.cuda.max_memory_reserved(device) / (1024**3)}")
        torch.cuda.reset_max_memory_allocated(device)

def plot_mem_stats(device):
    # Initialize lists to store data
    timestamps = []
    device_memory_used = []
    memory_allocated = []
    memory_reserved = []

    # Read the file and parse the data
    with open(MEMSTATS_OUPUT_FILE, 'r') as f:
        for line in f:
            # Split line based on '::' separator
            parts = line.strip().split('::')
            device_id = parts[0]
            ts = parts[1]
            label = parts[2]
            metric_type = parts[3].split(":")[0]
            if metric_type not in ['device_memory_used', 'memory_allocated', 'memory_reserved']:
                continue
            value = float(parts[4])

            # Filter out lines that don't match the target device ID
            if device_id != device:
                continue

            # Convert timestamp to datetime object for plotting
            timestamp = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
            timestamps.append(timestamp)

            # Add the value to the appropriate list
            if metric_type == 'device_memory_used':
                device_memory_used.append(value)
            elif metric_type == 'memory_allocated':
                memory_allocated.append(value)
            elif metric_type == 'memory_reserved':
                memory_reserved.append(value)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, device_memory_used, label='Device Memory Used (GB)', color='red')
    plt.plot(timestamps, memory_allocated, label='Memory Allocated (GB)', color='blue')
    plt.plot(timestamps, memory_reserved, label='Memory Reserved (GB)', color='green')

    # Customize the plot
    plt.xlabel('Timestamp')
    plt.ylabel('Memory (GB)')
    plt.title(f'GPU Memory Usage for Device ID {device} Over Time')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(f"gpu_memory_usage_device_{device}.png")
