import os
import pandas as pd
import matplotlib.pyplot as plt
from process_log import RESULTS_DIR, OUTPUT_DIR, CONF_EXPERIMENTS

# ---------- PART 2: Plotting ----------
def load_csv_results(results_dir=RESULTS_DIR, exp_list=CONF_EXPERIMENTS):
    """
    Load all CSV result files into a dictionary: {algo_name: DataFrame}.
    """
    all_algo_perf_data = {}
    for exp in exp_list:
        exp_path = os.path.join(results_dir, exp)
        if not os.path.isdir(exp_path):
            print(f"There is no result directories found at {exp_path}")
            continue
        for file in os.listdir(exp_path):
            if file.endswith(".csv"):
                filepath = os.path.join(exp_path, file)
                # Expected format: exp_algo_results.csv
                base = file.replace("_results.csv", "")
                try:
                    exp, algo = base.split("_", 1)
                except ValueError:
                    algo = base
                    exp = "default"

                df = pd.read_csv(filepath)
                df["algo"] = algo
                df["exp"] = exp
                all_algo_perf_data[algo] = df
    return all_algo_perf_data

def plot_results(data):
    """
    Plot accuracy, loss, and training time comparisons across algorithms.
    """
    metrics = ["accuracy", "loss", "train_time", "time_per_round_s", "time_minmax"]
    titles = {
        "accuracy": "Accuracy vs Number of Clients",
        "loss": "Loss vs Number of Clients",
        "train_time": "Training Time vs Number of Clients",
        "time_per_round_s": "Normalized Training Time vs Number of Clients",
        "time_minmax": "Normalized Training Time vs Number of Clients",
    }
    ylabels = {
        "accuracy": "Accuracy (%)",
        "loss": "Loss",
        "train_time": "Time (s)",
        "time_per_round_s": "Time/Iter",
        "time_minmax": "Time",
    }
    exp = "iid"
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        if data != {}:
            for algo_name, df in data.items():
                exp = df["exp"][0]
                df["time_per_round_s"] = df["train_time"] / 100  # or 150
                # Min–max across all algos/experiments
                all_times = pd.concat(list(data.values()))["train_time"]
                t_min, t_max = all_times.min(), all_times.max()
                df["time_minmax"] = (df["train_time"] - t_min) / (t_max - t_min + 1e-9)
                if metric not in df.columns:
                    continue
                df_sorted = df.sort_values("num_clients")
                # Per-round
                plt.plot(df_sorted["num_clients"], df_sorted[metric],
                         marker="o", label=algo_name)
        else:
            raise ValueError("no data extracted")

        plt.title(titles[metric])
        plt.xlabel("Number of Clients")
        plt.ylabel(ylabels[metric])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{exp}", f"{metric}_comparison.png"))
        plt.close()
