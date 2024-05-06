#!/bin/bash

# RUN it from the root folder of the project
decpy_path=./eval # Path to eval folder
run_path=./eval/data # Path to the folder where the graph and config file will be copied and the results will be stored
prefix_dir=new_exp_dpsgd_$(date '+%Y-%m-%dT%H:%M')
python_bin=./.venv/decentralizepy_env/bin
env_python=$python_bin/python
script_path=./tutorial/DPSGDwithNonIID # Path to the folder where the run_IDCAwPS.sh is located
eval_file=$decpy_path/testingDPSGDnIID.py # decentralized driver code (run on each machine)

machines=1 # number of machines in the runtime
iterations=80
test_after=4
log_level=INFO #INFO # DEBUG | INFO | WARN | CRITICAL
procs_per_machine=20

m=0 # machine id corresponding consistent with ip.json

echo "All started at $(date '+%Y-%m-%dT%H:%M')!"

config_file=config_new_exp_labels_dpsgd.ini
config_file_path=$script_path/configs/$config_file
configs=(1 2 3)
seeds=(1122 3344 5566 7788) # (12 34 56 78)

for config in ${configs[@]}
do
    echo "Config $config"
    case $config in
        1)
            $python_bin/crudini --set $config_file_path DATASET sizes "[[1/20]*10,[1/20]*10]"
            ;;
        2)
            $python_bin/crudini --set $config_file_path DATASET sizes "[[1/20]*15,[1/20]*5]"
            ;;
        3)
            $python_bin/crudini --set $config_file_path DATASET sizes "[[1/20]*18,[1/20]*2]"
            ;;
    esac

    cp $config_file_path $run_path

    echo M is $m
    echo procs per machine is $procs_per_machine

    for seed in ${seeds[@]}
    do
        echo "Seed $seed"
        $python_bin/crudini --set $run_path/$config_file DATASET random_seed $seed
        $python_bin/crudini --set $run_path/$config_file NODE graph_seed $(($seed+ 666))


        log_dir=$run_path/$prefix_dir/config$config/$(date '+%Y-%m-%dT%H:%M')/machine$m # in the eval folder
        echo $log_dir
        mkdir -p $log_dir

        $env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -ta $test_after -cf $run_path/$config_file -ll $log_level -wsd $log_dir
    done
done
echo "All done at $(date '+%Y-%m-%dT%H:%M')!"