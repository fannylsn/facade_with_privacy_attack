import logging
from pathlib import Path
from shutil import copy

from decentralizepy import utils
from decentralizepy.mappings.Linear import Linear
from decentralizepy.node.DPSGDNodeFederatedIFCA import DPSGDNodeFederatedIFCA
from decentralizepy.node.FederatedParameterServerIFCA import (
    FederatedParameterServerIFCA,
)
from localconfig import LocalConfig
from torch import multiprocessing as mp


def read_ini(file_path):
    config = LocalConfig(file_path)
    for section in config:
        print("Section: ", section)
        for key, value in config.items(section):
            print((key, value))
    print(dict(config.items("DATASET")))
    return config


if __name__ == "__main__":
    args = utils.get_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    log_level = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    config = read_ini(args.config_file)
    my_config = dict()
    for section in config:
        my_config[section] = dict(config.items(section))

    copy(args.config_file, args.log_dir)
    utils.write_args(args, args.log_dir)

    n_machines = args.machines
    procs_per_machine = args.procs_per_machine[0]
    l_mapping = Linear(n_machines, procs_per_machine)
    m_id = args.machine_id

    sm = args.server_machine
    sr = args.server_rank

    import os
    import sys

    # print(sys.path)

    # # Get the directory of the current script
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # print(script_dir)
    # # Search for the temporary directory containing the copied files
    # temp_dir = None
    # for dirname in os.listdir(script_dir):
    #     if dirname.startswith("tmp."):  # Assuming the temporary directory starts with "tmp."
    #         temp_dir = os.path.join(script_dir, dirname)
    #         break

    # if not temp_dir:
    #     # Assuming the 'src' directory is where the copied decentralizepy package resides
    #     src_dir = os.path.join(temp_dir, "src")

    #     # Insert the 'src' directory at the beginning of sys.path
    #     sys.path.insert(0, src_dir)

    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Traverse up the directory hierarchy until we find a directory that starts with "tmp."
        parent_dir = script_dir
        while not os.path.basename(parent_dir).startswith("tmp."):
            parent_dir = os.path.dirname(parent_dir)

            # If we reached the root directory and couldn't find the temporary directory, exit the loop
            if parent_dir == os.path.dirname(parent_dir):
                raise FileNotFoundError("Temporary directory not found.")

        # Construct the path to the temporary directory
        temp_dir = parent_dir

        # Assuming the 'src' directory is where the copied decentralizepy package resides
        src_dir = os.path.join(temp_dir, "src")

        # Insert the 'src' directory at the beginning of sys.path
        # sys.path.insert(0, src_dir)
        sys.path.append(src_dir)

    except FileNotFoundError:
        print("Normal run")

    while True:
        # dict to handle early restart
        manager = mp.Manager()
        return_dict = manager.dict()
        return_dict["early_stop"] = False

        processes = []
        if sm == m_id:
            processes.append(
                mp.Process(
                    target=FederatedParameterServerIFCA,
                    args=[
                        sr,
                        m_id,
                        l_mapping,
                        my_config,
                        args.iterations,
                        args.log_dir,
                        args.weights_store_dir,
                        log_level[args.log_level],
                        args.test_after,
                        args.train_evaluate_after,
                        args.working_rate,  # bit weird to call that a rate, its a ratio
                        return_dict,
                    ],
                )
            )

        for r in range(0, procs_per_machine):
            processes.append(
                mp.Process(
                    target=DPSGDNodeFederatedIFCA,
                    args=[
                        r,
                        m_id,
                        l_mapping,
                        my_config,
                        args.iterations,
                        args.log_dir,
                        args.weights_store_dir,
                        log_level[args.log_level],
                        args.test_after,
                        args.train_evaluate_after,
                        args.reset_optimizer,
                        sr,
                    ],
                )
            )

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        if not return_dict["early_stop"]:
            print("Normal stop")
            break

        # new seed because of fail to settle
        my_config["DATASET"]["random_seed"] = int(my_config["DATASET"]["random_seed"]) + 1
        print("Early restart, new seed is: ", my_config["DATASET"]["random_seed"])
