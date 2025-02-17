from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats
from tqdm import trange


def load_run_data(base_path: Path, n_unfrozen: int) -> dict:
    """Load and process data for a specific n_unfrozen_layers setting."""
    run_path = base_path / f"1nodes_16cpus_8gpus_{n_unfrozen}unfrozen"

    # Load config data
    config_data = {}
    try:
        for config_dir in (run_path / "configs").iterdir():
            id = config_dir.name.split("_")[-1]
            with open(config_dir / "report.yaml", "r") as f:
                report = yaml.safe_load(f)
            config_data[id] = {
                "loss": report["loss"],
                # "val_acc": report["full_fidelity_results"]["val_acc"],
                "full_fidelity_loss": report["extra"]["info_dict"]["full_fidelity_results"][
                    "val_err"
                ],
            }
    except Exception as e:
        print(f"Error loading config data for {run_path}: {e}")

    return config_data


def analyze_rank_correlation(
    base_path: Path, max_unfrozen: int = 10, use_full_fidelity: bool = False
):
    """Analyze rank correlation between different n_unfrozen_layers settings."""

    # key = "full_fidelity_loss" if use_full_fidelity else "loss"
    # print(f"Using {key} as the metric for rank correlation")
    # Store data for each n_unfrozen setting
    data_dict = {}
    for n in trange(1, max_unfrozen + 1):
        try:
            data_dict[n] = load_run_data(base_path, n)
        except FileNotFoundError:
            print(f"Data for n_unfrozen={n} not found")
            continue

    # Convert to DataFrame with config_id as index and n_unfrozen as columns
    # split into two dataframes one for loss one for full fidelity loss
    loss_data = {n: {id: data["loss"] for id, data in data_dict[n].items()} for n in data_dict}
    full_fidelity_loss_data = {
        n: {id: data["full_fidelity_loss"] for id, data in data_dict[n].items()} for n in data_dict
    }

    loss_data = pd.DataFrame.from_dict(loss_data)
    loss_data.index.name = "config_id"
    loss_data.columns.name = "n_unfrozen_layers"

    full_fidelity_loss_data = pd.DataFrame.from_dict(full_fidelity_loss_data)
    full_fidelity_loss_data.index.name = "config_id"
    full_fidelity_loss_data.columns.name = "n_unfrozen_layers"

    # print(all_data)

    # save to csv
    loss_data.to_csv(base_path / "../.." / f"loss_data_use_full_fidelity_{use_full_fidelity}.csv")
    full_fidelity_loss_data.to_csv(
        base_path / "../.." / f"full_fidelity_loss_data_use_full_fidelity_{use_full_fidelity}.csv"
    )

    # Calculate correlation matrix
    n_settings = len(loss_data.columns)
    corr_matrix = np.zeros((n_settings, n_settings))

    for i, n1 in enumerate(loss_data.columns):
        for j, n2 in enumerate(loss_data.columns):
            # Calculate Spearman correlation
            correlation, _ = stats.spearmanr(loss_data[n1], loss_data[n2])
            corr_matrix[i, j] = correlation

    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="RdYlBu",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=list(loss_data.columns),
        yticklabels=list(loss_data.columns),
    )
    plt.title("Spearman Rank Correlation of HP Configurations\nAcross n_unfrozen_layers Settings")
    plt.xlabel("n_unfrozen_layers")
    plt.ylabel("n_unfrozen_layers")

    # Save plot
    plt.savefig(base_path / "../.." / "rank_correlation_heatmap.png")
    plt.close()

    # Analyze top-K overlap
    k_values = [1, 5, 10]
    overlap_results = []

    for k in k_values:
        for n1 in loss_data.columns:
            for n2 in loss_data.columns:
                if n1 >= n2:
                    continue

                # Get top-K configs for each setting
                top_k1 = set(loss_data[n1].nsmallest(k).index)
                top_k2 = set(loss_data[n2].nsmallest(k).index)

                # Calculate overlap
                overlap = len(top_k1.intersection(top_k2))
                overlap_results.append(
                    {
                        "n_unfrozen_1": n1,
                        "n_unfrozen_2": n2,
                        "k": k,
                        "overlap": overlap,
                        "overlap_percentage": (overlap / k) * 100,
                    }
                )

    overlap_df = pd.DataFrame(overlap_results)
    overlap_df.to_csv(base_path / "top_k_overlap.csv", index=False)

    # Create line plot showing how top 10 configs from n_unfrozen=10 perform across different settings
    plt.figure(figsize=(10, 6))

    n_configs = 20
    # Get top 10 configs from n_unfrozen=10 setting
    n10_performance = full_fidelity_loss_data[10]  # Get performance when n_unfrozen=10
    top_n_configs = n10_performance.nsmallest(n_configs).index

    # Plot performance line for each top config
    for config in top_n_configs:
        plt.plot(
            loss_data.columns,
            loss_data.loc[config],
            marker="o",
            label=f"Config {config}",
        )

    plt.xlabel("Number of Unfrozen Layers")
    plt.ylabel("Validation Error")
    plt.title(
        f"Performance of Top {n_configs} Configs (Best at n_unfrozen=10)\nAcross Different Layer Freezing Settings"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save plot
    plt.savefig(
        base_path / "../.." / f"top_{n_configs}_configs_performance_from_n10.png",
        bbox_inches="tight",
    )
    plt.close()

    # Create line plot showing ranks of top 10 configs across different settings
    plt.figure(figsize=(10, 6))

    # Get performance data just for top 10 configs
    top_n_performance = loss_data.loc[top_n_configs]

    # Calculate ranks among just the top 10 configs for each n_unfrozen setting
    ranks = top_n_performance.rank()

    # Plot rank line for each top config
    plt.figure(figsize=(10, 6))
    for config in top_n_configs:
        plt.plot(
            loss_data.columns, ranks.loc[config], marker="o", label=f"Config {config}",
        )

    plt.xlabel("Number of Unfrozen Layers")
    plt.ylabel(f"Rank Among Top {n_configs}")
    plt.title(
        f"Relative Rank of Top {n_configs} Configs (Best at n_unfrozen=10)\nAcross Different Layer Freezing Settings"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.gca().invert_yaxis()  # Invert y-axis so rank 1 is at the top

    # Save plot
    plt.savefig(
        base_path / "../.." / f"top_{n_configs}_configs_relative_ranks_from_n10.png",
        bbox_inches="tight",
    )
    plt.close()

    # Get performance data just for top 10 configs
    top_n_performance = full_fidelity_loss_data.loc[top_n_configs]

    # Calculate ranks among just the top 10 configs for each n_unfrozen setting
    ranks = top_n_performance.rank()

    # Plot average rank across settings for each config
    plt.figure(figsize=(10, 6))
    avg_ranks = ranks.mean(axis=1).sort_values()

    # Get ranks at n_unfrozen_layers = 10
    ranks_at_10 = ranks[10]

    # Create bar plot of average ranks
    bars = plt.bar(range(len(avg_ranks)), avg_ranks)

    # Add markers for rank at n=10 layers
    plt.plot(
        range(len(avg_ranks)), [ranks_at_10[i] for i in avg_ranks.index], "ro", label="Rank at n=10"
    )

    plt.xticks(range(len(avg_ranks)), [f"Config {i}" for i in avg_ranks.index], rotation=45)
    plt.xlabel("Configuration")
    plt.ylabel("Rank")
    plt.title(
        f"Average Rank of Top {n_configs} Configs Across All Layer Freezing Settings\n(Red dots show rank at n=10)"
    )
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(
        base_path / "../.." / f"top_{n_configs}_configs_average_ranks.png", bbox_inches="tight"
    )
    plt.close()

    return corr_matrix, overlap_df


if __name__ == "__main__":
    # Assuming the output directory structure from the original script
    repo_root = Path(__file__).parent.parent
    base_path = repo_root / "output" / "resnet18_hp_grid" / "GridSearch"

    # corr_matrix, overlap_df = analyze_rank_correlation(base_path)
    analyze_rank_correlation(base_path)

    # Print summary statistics
    # print(
    #     "\nAverage rank correlation:", np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
    # )
    # print("\nTop-K overlap summary:")
    # print(overlap_df.groupby("k")["overlap_percentage"].describe())
