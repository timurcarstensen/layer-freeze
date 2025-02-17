import glob
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def extract_unfrozen_layers(path: str) -> int:
    """Extract number of unfrozen layers from the run path."""
    match = re.search(r"(\d+)unfrozen", path)
    return int(match.group(1)) if match else -1


def read_tensorboard_data(log_dir: str) -> List[Tuple[int, float, float]]:
    """
    Read tensorboard logs and extract timing information.

    Returns:
        List of tuples (unfrozen_layers, avg_forward_time, avg_backward_time)
    """
    results: List[Tuple[int, float, float]] = []
    print(f"Reading tensorboard data from {os.path.abspath(log_dir)}")
    event_files = []
    for dirpath, dirnames, filenames in os.walk(log_dir):
        for filename in filenames:
            if filename.startswith("events.out.tfevents"):
                event_files.append(os.path.join(dirpath, filename))

    print(f"Found {len(event_files)} event files")
    # for event_file in event_files:
    #     print(f"  {event_file}")

    # Find all event files recursively
    for event_file in event_files:
        run_dir = os.path.dirname(event_file)
        unfrozen_layers = extract_unfrozen_layers(run_dir)

        if unfrozen_layers == -1:
            continue

        # Load the event file
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        config_name = ea.Tags()["scalars"][0].split("/")[0]

        # Extract timing data
        try:
            forward_times = [
                event.value for event in ea.Scalars(f"{config_name}/avg_forward_time_ms")
            ]
            backward_times = [
                event.value for event in ea.Scalars(f"{config_name}/avg_backward_time_ms")
            ]
            full_fidelity_val_acc = [
                event.value for event in ea.Scalars(f"{config_name}/full_fidelity_val_acc")
            ]
            full_fidelity_val_err = [
                event.value for event in ea.Scalars(f"{config_name}/full_fidelity_val_err")
            ]

            # Calculate averages
            avg_forward = np.mean(forward_times) if forward_times else 0
            avg_backward = np.mean(backward_times) if backward_times else 0
            avg_full_fidelity_val_acc = (
                np.mean(full_fidelity_val_acc) if full_fidelity_val_acc else 0
            )
            avg_full_fidelity_val_err = (
                np.mean(full_fidelity_val_err) if full_fidelity_val_err else 0
            )
            results.append(
                (
                    unfrozen_layers,
                    avg_forward,
                    avg_backward,
                    avg_full_fidelity_val_acc,
                    avg_full_fidelity_val_err,
                )
            )
        except KeyError:
            continue

    df = pd.DataFrame(
        results,
        columns=[
            "n_unfrozen_layers",
            "avg_forward_time_ms",
            "avg_backward_time_ms",
            "full_fidelity_val_acc",
            "full_fidelity_val_err",
        ],
    )
    return df


def plot_timing_analysis(
    # full_fidelity_val_accs: List[float],
    # full_fidelity_val_errs: List[float],
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """Create and save the timing analysis plot."""
    if df.empty:
        print("No data found to plot!")
        return

    unfrozen_layers = df.index.to_list()
    forward_times = df["avg_forward_time_ms"]
    backward_times = df["avg_backward_time_ms"]
    # full_fidelity_val_acc = df["full_fidelity_val_acc"]
    # full_fidelity_val_err = df["full_fidelity_val_err"]

    plt.figure(figsize=(10, 6))
    plt.plot(unfrozen_layers, forward_times, "b-o", label="Forward Time")
    plt.plot(unfrozen_layers, backward_times, "r-o", label="Backward Time")
    # plt.plot(unfrozen_layers, full_fidelity_val_accs, "g-o", label="Full Fidelity Val Acc")
    # plt.plot(unfrozen_layers, full_fidelity_val_errs, "m-o", label="Full Fidelity Val Err")
    plt.xlabel("Number of Unfrozen Layers")
    plt.ylabel("Average Time (seconds)")
    plt.title("Forward and Backward Pass Times vs. Number of Unfrozen Layers")
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig(output_path)
    plt.close()


def get_full_fidelity_perf_of_best_model(
    run_status_df: pd.DataFrame, config_data: pd.DataFrame, n_unfrozen_layers: int
) -> Tuple[float, float]:
    """Get the full fidelity performance of the best model."""
    best_model_idx = int(run_status_df.loc["best_config_id"][0])
    best_model_config = config_data.loc[best_model_idx]
    full_fidelity_val_acc = best_model_config["result.info_dict.full_fidelity_results.val_acc"]
    full_fidelity_val_err = best_model_config["result.info_dict.full_fidelity_results.val_err"]
    low_fidelity_val_acc = best_model_config["result.loss"]
    return full_fidelity_val_acc, full_fidelity_val_err, low_fidelity_val_acc


def get_all_summary_csv_dfs(log_dir: str) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Get all summary csv dfs."""
    n_unfrozen_layers = [1, 2, 3, 4, 5, 6, 7, 8]
    results = {n: {"run_status_df": None, "config_df": None} for n in n_unfrozen_layers}
    for dirpath, _, _ in os.walk(log_dir):
        if any(f"{n}unfrozen" in dirpath for n in range(1, 9)):
            n_unfrozen_layers = extract_unfrozen_layers(dirpath)
            if results[n_unfrozen_layers]["run_status_df"] is None:
                results[n_unfrozen_layers]["run_status_df"] = pd.read_csv(
                    os.path.join(dirpath, "summary_csv", "run_status.csv"), index_col=0
                )
                results[n_unfrozen_layers]["config_df"] = pd.read_csv(
                    os.path.join(dirpath, "summary_csv", "config_data.csv")
                )
    return results


def main() -> None:
    log_dir = "./output"
    output_path = "./output/timing_analysis.png"

    # Read data from tensorboard logs
    df = read_tensorboard_data(log_dir)
    results = get_all_summary_csv_dfs(log_dir)
    full_fidelity_val_accs = []
    full_fidelity_val_errs = []
    low_fidelity_val_accs = []
    for k, v in results.items():
        run_status_df = v["run_status_df"]
        config_df = v["config_df"]
        full_fidelity_val_acc, full_fidelity_val_err, low_fidelity_val_acc = (
            get_full_fidelity_perf_of_best_model(run_status_df, config_df, k)
        )
        full_fidelity_val_accs.append(full_fidelity_val_acc)
        full_fidelity_val_errs.append(full_fidelity_val_err)
        low_fidelity_val_accs.append(low_fidelity_val_acc)

    # Plot full fidelity validation accuracy vs number of unfrozen layers
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 9), full_fidelity_val_errs, marker="o")
    plt.xlabel("Number of Unfrozen Layers")
    plt.ylabel("Full Fidelity Validation Accuracy (%)")
    plt.title("Full Fidelity Validation Accuracy vs Number of Unfrozen Layers")
    plt.grid(True)
    plt.savefig("./output/full_fidelity_acc_vs_layers.png")
    plt.close()
    # Create and save the plot
    plot_timing_analysis(df=df, output_path=output_path)

    # Print the data
    print("\nTiming Analysis Data:")
    print(df)


if __name__ == "__main__":
    main()
