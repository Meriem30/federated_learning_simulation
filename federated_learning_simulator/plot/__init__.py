from process_log import *
from ploter import *



if __name__ == "__main__":
    #algo_data = process_logs()
    algo_data = load_csv_results()
    plot_results(algo_data)
    print(f"Results saved")