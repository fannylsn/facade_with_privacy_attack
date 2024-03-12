import logging
from pathlib import Path
from shutil import copy

from decentralizepy import utils
from decentralizepy.mappings.Linear import Linear
from decentralizepy.node.DPSGDNodeIDCAwPS import DPSGDNodeIDCAwPS
from localconfig import LocalConfig
from torch import multiprocessing as mp


def read_ini(file_path):
    config = LocalConfig(file_path)
    for section in config:
        print("Section: ", section)
        for key, value in config.items(section):
            print((key, value))
    print(file_path)
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

    l_maping = Linear(n_machines, procs_per_machine)
    m_id = args.machine_id

    processes = []
    for r in range(procs_per_machine):
        processes.append(
            mp.Process(
                target=DPSGDNodeIDCAwPS,
                args=[
                    r,
                    m_id,
                    l_maping,
                    my_config,
                    args.iterations,
                    args.log_dir,
                    args.weights_store_dir,
                    log_level[args.log_level],
                    args.test_after,
                    args.train_evaluate_after,
                    args.reset_optimizer,
                ],
            )
        )

    # spawn processes
    for p in processes:
        p.start()

    for p in processes:
        p.join()
