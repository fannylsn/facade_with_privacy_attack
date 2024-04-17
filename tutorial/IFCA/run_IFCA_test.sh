#!/bin/bash

# Get the absolute path of the current directory
current_dir=$(pwd)

# Temporary directory to store copied codebase
temp_dir=$(mktemp -d)

# List of directories to exclude from copying
exclude_venv=".venv"
excluse_data="eval/data"
exclude_past_data="eval/all_past_exp"

# Copy necessary files from codebase to temporary directory
rsync -av --exclude=$exclude_venv --exclude=$excluse_data --exclude=$exclude_past_data ./ "$temp_dir/"
rsync -av ./eval/data/cifar-10-batches-py "$temp_dir/eval/data/"

# Absolute path to the folder where the results will be stored
run_path=$current_dir/eval/data
env_python=$current_dir/.venv/decentralizepy_env/bin/python

# Paths within the temporary directory
decpy_path=$temp_dir/eval
script_path=$temp_dir/tutorial/IFCA

# Experiment configurations
prefix_dir=experiment_minority_ifca_$(date '+%Y-%m-%dT%H:%M')
machines=1
# 80
iterations=2
#4
test_after=1
log_level=INFO
server_rank=-1
server_machine=0
working_rate=1
m=0

echo "All started at $(date '+%Y-%m-%dT%H:%M')!"

# Change working directory to temporary directory
cd "$temp_dir"

for config in 1 2 3 4; do
    case $config in
        1)
            config_file=config_1to1_ifca.ini
            procs_per_machine=32 ;;
        2)
            config_file=config_1to2_ifca.ini
            procs_per_machine=24 ;;
        3)
            config_file=config_1to4_ifca.ini
            procs_per_machine=20 ;;
        4)
            config_file=config_1to8_ifca.ini
            procs_per_machine=18 ;;
    esac

    cp $decpy_path/step_configs/$config_file $run_path

    echo "Config: $config, procs per machine: $procs_per_machine"

    for seed in 111 222 333 444 555; do
        field_name=random_seed
        sed -i "s/\($field_name *= *\).*/\1$seed/" $run_path/$config_file

        log_dir=$run_path/$prefix_dir/config$config/$(date '+%Y-%m-%dT%H:%M')/machine$m
        echo "Log directory: $log_dir"
        mkdir -p $log_dir

        echo "file: $decpy_path/testingIFCA.py"

        $env_python $decpy_path/testingIFCA.py \
            -ro 0 \
            -tea $test_after \
            -ld $log_dir \
            -wsd $log_dir \
            -mid $m \
            -ps $procs_per_machine \
            -ms $machines \
            -is $iterations \
            -ta $test_after \
            -cf $run_path/$config_file \
            -ll $log_level \
            -sm $server_machine \
            -sr $server_rank \
            -wr $working_rate
    done
done

# Cleanup: Return to the original directory
cd "$current_dir" && rm -rf "$temp_dir"

echo "All done at $(date '+%Y-%m-%dT%H:%M')!"