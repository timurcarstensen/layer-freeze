import argparse
import gc
import os

import git
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from layer_freeze.model_agnostic_freezing import FrozenModel
from layer_freeze.utils import (
    ModelPerfStats,
    measure_model_training_hw_metrics,
)

if __name__ == "__main__":
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
    for n_trainable in range(1, 16):
        model = FrozenModel(
            n_trainable=n_trainable,
            base_model=torchvision.models.vit_b_16(weights=None),
            print_summary=True,
            unwrap=torchvision.models.vision_transformer.Encoder,
        )
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
        )
        stats.append(
            measure_model_training_hw_metrics(
                model, dataloader, optimizer, device="cuda", warmup_passes=30
            )
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

    repo = git.Repo(".", search_parent_directories=True)
    base_path = os.path.join(
        repo.working_tree_dir, "results", "model_perf", args.base_model, "quantized_frozen_layers"
    )
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
