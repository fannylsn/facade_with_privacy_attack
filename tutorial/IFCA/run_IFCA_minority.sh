#!/bin/bash

# RUN it from the root folder of the project
decpy_path=./eval # Path to eval folder
run_path=./eval/data # Path to the folder where the graph and config file will be copied and the results will be stored
prefix_dir=experiment_minority_ifca_$(date '+%Y-%m-%dT%H:%M')
env_python=./.venv/decentralizepy_env/bin/python

script_path=./tutorial/IFCA # Path to the folder where the run_IDCAwPS.sh is located

machines=1 # number of machines in the runtime
iterations=80
test_after=4
# CAREFULL restarts
eval_file=$decpy_path/testingIFCA_restarts.py # decentralized driver code (run on each machine)
log_level=INFO #INFO # DEBUG | INFO | WARN | CRITICAL

server_rank=-1
server_machine=0
working_rate=1

m=0 # machine id corresponding consistent with ip.json

echo "All started at $(date '+%Y-%m-%dT%H:%M')!"


for config in 4 3 2 1
do
    if [ $config -eq 1 ]
    then
        config_file=config_1to1_ifca.ini
        procs_per_machine=32 #16 vs 16
    elif [ $config -eq 2 ]
    then
        config_file=config_1to2_ifca.ini
        procs_per_machine=24 #16 vs 8
    elif [ $config -eq 3 ]
    then
        config_file=config_1to4_ifca.ini
        procs_per_machine=20 #16 vs 4
    elif [ $config -eq 4 ]
    then
        config_file=config_1to8_ifca.ini
        procs_per_machine=18 #16 vs 2
    fi

    cp $decpy_path/step_configs/$config_file $run_path

    echo M is $m
    echo procs per machine is $procs_per_machine

    for seed in 11 22 # 33 44 55
    do
        field_name=random_seed
        sed -i "s/\($field_name *= *\).*/\1$seed/" $run_path/$config_file

        log_dir=$run_path/$prefix_dir/config$config/$(date '+%Y-%m-%dT%H:%M')/machine$m # in the eval folder
        echo $log_dir
        mkdir -p $log_dir

        $env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -wsd $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -ta $test_after -cf $run_path/$config_file -ll $log_level -sm $server_machine -sr $server_rank -wr $working_rate
    done
done

echo "All done at $(date '+%Y-%m-%dT%H:%M')!"