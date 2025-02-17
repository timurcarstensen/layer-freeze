import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from layer_freeze.freeze_layers import freeze_layers


def measure_timings(model, dataloader, device, n_forward_passes=3) -> Tuple[float, float, float]:
    optimizer_adam = Adam(model.parameters(), lr=0.001)
    tmp_forward_time = []
    tmp_backward_time = []
    for _ in range(n_forward_passes):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_adam.zero_grad()

            start_time = time.time()
            outputs = model(inputs)
            tmp_forward_time.append(time.time() - start_time)

            loss = torch.nn.CrossEntropyLoss()(outputs, labels)

            start_time = time.time()
            loss.backward()
            tmp_backward_time.append(time.time() - start_time)

            optimizer_adam.step()

    return (
        np.mean(tmp_forward_time),
        np.mean(tmp_backward_time),
        torch.cuda.memory_allocated(device) / (1024**2),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_layers", type=int, default=53)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--n_forward_passes", type=int, default=3)
    args = parser.parse_args()
    max_layers = args.max_layers
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    full_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataset = torch.utils.data.Subset(full_dataset, range(1000))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=False,
        prefetch_factor=1,
        persistent_workers=False,
    )

    memory_usage = []
    perc_frozen = []
    backward_time = []
    forward_time = []
    for i in tqdm(range(1, max_layers)):
        model = models.resnet18(weights=None)
        frozen_params, total_params = freeze_layers(model, i, return_param_stats=True)
        perc_frozen.append(frozen_params / total_params)
        model = model.to(device)
        optimizer_adam = Adam(model.parameters(), lr=0.001)

        for _ in range(args.n_forward_passes):
            _ = measure_timings(
                model, dataloader, device, args.n_forward_passes
            )

        fwd_time, bwd_time, mem_usage = measure_timings(
            model, dataloader, device, args.n_forward_passes
        )
        forward_time.append(fwd_time)
        backward_time.append(bwd_time)
        memory_usage.append(mem_usage)

        del model
        del optimizer_adam

    # Plot percentage of frozen parameters vs memory usage
    plt.figure(figsize=(10, 6))
    plt.plot(perc_frozen, memory_usage, "b-", label="Memory Usage (MB)")
    plt.xlabel("Percentage of Frozen Parameters")
    plt.ylabel("Memory (MB)")
    plt.title(f"Memory Usage vs Percentage of Frozen Parameters (Batch Size: {batch_size})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"memory_usage_plot_batch_{batch_size}.png", dpi=300)
    plt.close()

    # Plot number of unfrozen layers vs memory usage
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_layers), memory_usage, "r-", label="Memory Usage (MB)")
    plt.xlabel("Number of Unfrozen Layers")
    plt.ylabel("Memory (MB)")
    plt.title(f"Memory Usage vs Number of Unfrozen Layers (Batch Size: {batch_size})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"memory_usage_layers_plot_batch_{batch_size}.png", dpi=300)
    plt.close()

    # Plot forward and backward time vs number of unfrozen layers
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_layers), forward_time, "g-", label="Forward Pass Time (ms)")
    plt.plot(range(1, max_layers), backward_time, "m-", label="Backward Pass Time (ms)")
    plt.ylim(0, 0.01)
    plt.xlabel("Number of Unfrozen Layers")
    plt.ylabel("Time (ms)")
    plt.title(
        f"Forward and Backward Pass Times vs Number of Unfrozen Layers (Batch Size: {batch_size})"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(f"pass_times_plot_batch_{batch_size}.png", dpi=300)
    plt.close()
