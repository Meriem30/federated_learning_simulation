import os
import re
import csv
import matplotlib.pyplot as plt
import pandas as pd
# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR = OUTPUT_DIR
LOG_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", "log"))
CONF_EXPERIMENTS = ["iid"]


os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regex patterns for extracting metrics
ROUND100_PATTERN = re.compile(r"round:\s*100.*loss:([\d.]+), accuracy:([\d.]+)%")
TIME_PATTERN = re.compile(r"training took\s*([\d.]+)\s*seconds")

# ---------- PART 1: Parse Logs ----------
def parse_log_file(filepath):
    """
    Extract accuracy, loss, and training time from a log file.
    """
    acc, loss, time_spent = None, None, None

    with open(filepath, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        block_size = 8192  # 1024, 4096, 8192
        data = b""
        while file_size > 0 and (acc is None or loss is None or time_spent is None):
            read_size = min(block_size, file_size)
            file_size -= read_size
            f.seek(file_size)
            data = f.read(read_size) + data
            lines = data.splitlines()

            for line in reversed(lines):
                line = line.decode("utf-8", errors="ignore")
                if "round: 100" in line and (acc is None or loss is None):
                    match = ROUND100_PATTERN.search(line)
                    if match:
                        loss = float(match.group(1))
                        acc = float(match.group(2))
                        last_round_performance = (loss, acc)
                elif "training took" in line and time_spent is None:
                    match = TIME_PATTERN.search(line)
                    if match:
                        train_time = float(match.group(1))

                if acc is not None and loss is not None and time_spent is not None:
                    break
    return last_round_performance, train_time

def process_logs(root=LOG_ROOT, exp_list=CONF_EXPERIMENTS):
    """
    Walk through directories and parse log into CSV files, one per algorithm.
    """
    algo_perf_data = {}
    for exp in exp_list:
        exp_path = os.path.join(root, exp)
        if not os.path.isdir(exp_path):
            print(f"There is no performance directories found at {exp_path}")
            continue

        for algo in os.listdir(exp_path):
            algo_path = os.path.join(exp_path, algo)
            if not os.path.isdir(algo_path):
                continue

            results = []
            for client_dir in os.listdir(algo_path):
                if "CLS" not in client_dir:
                    raise ValueError("No CLS in directory's name")

                match = re.search(r"(\d+)CLS$", client_dir)  # number followed by CLS at end
                if match:
                    num_clients = int(match.group(1))
                else:
                    raise ValueError(f"Could not extract client number from: {client_dir}")

                client_path = os.path.join(algo_path, client_dir)

                # get the single file inside
                for run_id in os.listdir(client_path):
                    log_file = os.path.join(client_path, run_id)
                    if os.path.isfile(log_file):
                        (loss, acc), time_spent = parse_log_file(log_file)
                        record = [num_clients, acc, loss, time_spent]
                        print(f"extracted record : {record}")
                        results.append(record)

            # sort by client number
            results.sort(key=lambda x: x[0])

            if results:
                df = pd.DataFrame(results)
                algo_perf_data[algo] = df
                print(algo_perf_data)
                # df.to_csv(os.path.join(OUTPUT_DIR, f"{algo}_results.csv"), index=False)
            else:
                raise ValueError(F"No results extracted for {algo}")

            # save CSV per algo per exp
            out_file = os.path.join(OUTPUT_DIR, f"{exp}_", f"{exp}_{algo}_results.csv")
            with open(out_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["num_clients", "accuracy", "loss", "train_time"])
                writer.writerows(results)

            print(f"[OK] Saved results for {algo} in {exp} → {out_file}")

    return algo_perf_data
