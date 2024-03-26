#!/bin/bash

# RUN it from the root folder of the project
decpy_path=./eval # Path to eval folder
run_path=./eval/data # Path to the folder where the graph and config file will be copied and the results will be stored
prefix_dir=experiment_minority$(date '+%Y-%m-%dT%H:%M')
env_python=./.venv/decentralizepy_env/bin/python

script_path=./tutorial/IDCAwithPeerSampling # Path to the folder where the run_IDCAwPS.sh is located

machines=1 # number of machines in the runtime
iterations=80
test_after=4
eval_file=$decpy_path/testingIDCAwPS.py # decentralized driver code (run on each machine)
log_level=INFO #INFO # DEBUG | INFO | WARN | CRITICAL

m=0 # machine id corresponding consistent with ip.json

echo "All started at $(date '+%Y-%m-%dT%H:%M')!"


for config in 1 2 3 4
do
    if [ $config -eq 1 ]
    then
        config_file=config_1to1.ini
        procs_per_machine=32 #16 vs 16
    elif [ $config -eq 2 ]
    then
        config_file=config_1to2.ini
        procs_per_machine=24 #16 vs 8
    elif [ $config -eq 3 ]
    then
        config_file=config_1to4.ini
        procs_per_machine=20 #16 vs 4
    elif [ $config -eq 4 ]
    then
        config_file=config_1to8.ini
        procs_per_machine=18 #16 vs 2
    fi

    cp $decpy_path/step_configs/$config_file $run_path

    echo M is $m
    echo procs per machine is $procs_per_machine

    for seed in 1111 2222 3333 4444 5555
    do
        field_name=random_seed
        sed -i "s/\($field_name *= *\).*/\1$seed/" $run_path/$config_file

        field_name=graph_seed
        graph_seed=$(($seed + 8888)) # offset random
        sed -i "s/\($field_name *= *\).*/\1$graph_seed/" $run_path/$config_file

        log_dir=$run_path/$prefix_dir/config$config/$(date '+%Y-%m-%dT%H:%M')/machine$m # in the eval folder
        echo $log_dir
        mkdir -p $log_dir

        $env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -ta $test_after -cf $run_path/$config_file -ll $log_level -wsd $log_dir
    done
done
echo "All done at $(date '+%Y-%m-%dT%H:%M')!"