from process_log import *
from ploter import *



if __name__ == "__main__":
    algo_data = load_csv_results()
    if not algo_data:  # If nothing was cached, parse logs
        print("No cached results found, parsing logs...")
        algo_data = process_logs()

    plot_results(algo_data)
    print(f"Results saved in {OUTPUT_DIR}")