import json
import os

import pandas as pd
import torch

from .session import Session


def dump_analysis() -> None:
    session_path = os.getenv("session_path")
    assert session_path is not None
    session_path = session_path.strip()
    session = Session(session_path)
    config = session.config
    res = {}
    res["exp_name"] = config.exp_name
    res["distributed_algorithm"] = config.distributed_algorithm
    res["dataset_name"] = config.dc_config.dataset_name
    res["model_name"] = config.model_config.model_name
    if config.endpoint_kwargs:
        res["endpoint_kwargs"] = config.endpoint_kwargs
    if config.trainer_config.dataloader_kwargs:
        res["dataloader_kwargs"] = config.trainer_config.dataloader_kwargs
    res["round"] = len(session.rounds)
    res["worker_number"] = config.worker_number
    if config.algorithm_kwargs:
        res |= config.algorithm_kwargs
    res["dataset_sampling"] = config.dataset_sampling
    if config.dataset_sampling_kwargs:
        res |= config.dataset_sampling_kwargs
    res["last_test_acc"] = session.last_test_acc
    res["mean_test_acc"] = session.mean_test_acc
    # initialize a dict with experiment name and dataset
    total_worker_cnts = {
        "exp_name": config.exp_name,
        "dataset_name": config.dc_config.dataset_name,
    }

    # analyse worker data (a dict for each worker in worker_data)
    for data in session.worker_data.values():
        # iterate over diff metric:value for each worker dict
        for k, v in data.items():
            # handle counting metric
            if "cnt" in k or "byte" in k:
                if k in ("embedding_bytes", "model_bytes"):
                    # append it directly and skip the rest
                    total_worker_cnts[k] = v
                    continue
                # for specific metrics
                if "edge_cnt" in k or "node_cnt" in k:
                    if k not in total_worker_cnts:
                        total_worker_cnts[k] = []
                    # append the value v for this metric to the list and skip the rest
                    total_worker_cnts[k].append(v)
                    continue
                # for other metric
                if k not in total_worker_cnts:
                    total_worker_cnts[k] = {}
                # iterate over the sub-dict v
                for k2, v2 in v.items():
                    if k2 not in total_worker_cnts[k]:
                        # initialize with the v2 if not already exist
                        total_worker_cnts[k][k2] = v2
                    else:
                        # otherwise, add v2 to the existing value
                        total_worker_cnts[k][k2] = total_worker_cnts[k][k2] + v2
    # calculate mean and standard deviation for edge/node metrics
    for k, v in total_worker_cnts.items():
        if "edge_cnt" in k or "node_cnt" in k:
            std, mean = torch.std_mean(torch.tensor(v, dtype=torch.float))
            # update with the computed mean and std
            total_worker_cnts[k] = {"mean": mean.item(), "std": std.item()}

    # merge results into res
    res |= total_worker_cnts
    res |= {"performance": session.round_record}
    for k, v in res.items():
        if isinstance(v, dict):
            # convert dict values in res to JSON str
            res[k] = json.dumps(v)
    # define the column list for the dataframe
    col_list = [
        "distributed_algorithm",
        "dataset_name",
        "model_name",
        "last_test_acc",
        "mean_test_acc",
        "round",
        "worker_number",
    ]

    # append any experiment or algorithm specific keys to the col_list
    if config.exp_name:
        col_list = ["exp_name"] + col_list
    for k in config.algorithm_kwargs:
        col_list.append(k)
    # extract the set of keys in res but not in col_list, append this list to the col_list
    col_list += list(set(res.keys()) - set(col_list))
    # create a dataframe from the res
    df = pd.DataFrame([res])
    # reorder columns to match col_list
    df = df[col_list]
    # df = df.drop_duplicates(ignore_index=True)
    # df = df.sort_values(by=col_list, ascending=False, ignore_index=True)
    # create the output file
    output_file = "exp.txt"
    # if already exist, read it into a dataframe
    if os.path.isfile(output_file):
        old_df = pd.read_csv(output_file)
        # concatenate the new and old dataframe
        df = pd.concat([old_df, df], ignore_index=True)
    # save the dataframe to multiple file format
    df.to_csv(output_file, index=False)
    df.to_excel("exp.xlsx", index=False, sheet_name="result")
    df.to_json("exp.json")
