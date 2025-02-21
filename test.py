import gc
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from layer_freeze.model_agnostic_freezing import FrozenModel


@dataclass
class ModelPerfStats:
    median_fwd_time: float
    std_fwd_time: float
    median_bwd_time: float
    std_bwd_time: float
    median_loop_time: float
    memory: float
    frac_trainable_params: float
    optimizer_time: float


def frac_trainable_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(
        p.numel() for p in model.parameters()
    )


def model_perf_stats(
    model: nn.Module,
    dataloader: "DataLoader",
    optimizer: torch.optim.Optimizer,
    warmup_passes: int = 100,
    device: torch.device | None = None,
) -> ModelPerfStats:
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)

    # perform warmup passes
    for i, batch in enumerate(tqdm(dataloader, desc="Warming up model", total=warmup_passes)):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i > warmup_passes:
            break

    fwd_times = []
    bwd_times = []
    loop_times = []
    optimizer_times = []
    max_memory_usage_recorded = 0
    for batch in tqdm(dataloader, desc="Measuring forward times"):
        loop_start = time.perf_counter()
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        fwd_time = time.perf_counter()
        pred = model(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        loss = loss.to(device)
        fwd_times.append(time.perf_counter() - fwd_time)

        bwd_time = time.perf_counter()
        loss.backward()
        bwd_times.append(time.perf_counter() - bwd_time)

        opt_start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad()
        optimizer_times.append(time.perf_counter() - opt_start)

        loop_times.append(time.perf_counter() - loop_start)
        max_memory_usage_recorded = max(
            max_memory_usage_recorded, torch.cuda.memory_allocated(device) / (1024**2)
        )

    assert torch.cuda.memory_allocated(device) / (1024**2) == max_memory_usage_recorded
    return ModelPerfStats(
        median_fwd_time=np.median(fwd_times),
        std_fwd_time=np.std(fwd_times),
        median_bwd_time=np.median(bwd_times),
        std_bwd_time=np.std(bwd_times),
        median_loop_time=np.median(loop_times),
        memory=max_memory_usage_recorded,
        frac_trainable_params=frac_trainable_params(model),
        optimizer_time=np.median(optimizer_times),
    )


if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="resnet152")
    parser.add_argument("--quantize_frozen_layers", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataset = torch.utils.data.Subset(dataset, range(100 * args.batch_size))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        persistent_workers=True,
    )

    stats: list[ModelPerfStats] = []
    for n_trainable in range(1, 10):
        model = FrozenModel(
            n_trainable=n_trainable,
            base_model=torchvision.models.vit_b_16(weights=None),
            print_summary=True,
            unwrap=torchvision.models.vision_transformer.Encoder,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        stats.append(
            model_perf_stats(model, dataloader, optimizer, device="cuda", warmup_passes=30)
        )
        del model
        del optimizer
        torch.cuda.empty_cache()
        gc.collect()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # Create second y-axis sharing same x-axis

    # Plot forward and backward times on left y-axis
    ax1.plot(
        [stat.frac_trainable_params for stat in stats],
        [stat.median_fwd_time for stat in stats],
        label="Forward Time",
        marker="o",
        linestyle="solid",
    )
    ax1.plot(
        [stat.frac_trainable_params for stat in stats],
        [stat.median_bwd_time for stat in stats],
        label="Backward Time",
        marker="o",
        linestyle="solid",
    )
    ax1.plot(
        [stat.frac_trainable_params for stat in stats],
        [stat.median_loop_time for stat in stats],
        label="Loop Time",
        marker="o",
        linestyle="--",
    )

    ax1.plot(
        [stat.frac_trainable_params for stat in stats],
        [stat.optimizer_time for stat in stats],
        label="Optimizer step + zero grad",
        marker="o",
        linestyle="--",
    )

    ax1.set_xlabel("Fraction of Trainable Parameters")
    ax1.set_ylabel("Time (s)")

    # Plot memory on right y-axis
    ax2.plot(
        [stat.frac_trainable_params for stat in stats],
        [stat.memory for stat in stats],
        marker="o",
        label="Memory",
        linestyle=":",
    )
    ax2.set_ylabel("Memory (MB)")

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    base_path = "./results/model_perf/vit/quantized_frozen_layers"
    os.makedirs(base_path, exist_ok=True)
    plt.grid(True, axis="both", linestyle="--", alpha=0.4, which="major")
    plt.title(
        f"Model Performance vs Trainable Parameters ({args.base_model})\n"
        f"{'Quantized Frozen Layers' if args.quantize_frozen_layers else 'Regular'} "
        f"(Batch Size: {args.batch_size})"
    )
    save_path = os.path.join(
        base_path,
        "quantized" if args.quantize_frozen_layers else "regular",
        f"model_perf_{args.base_model}_bs{args.batch_size}.pdf",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()

    # import pdb; pdb.set_trace()
