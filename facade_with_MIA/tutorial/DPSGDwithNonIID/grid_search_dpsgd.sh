#!/bin/bash

# RUN it from the root folder of the project
decpy_path=./eval # Path to eval folder
run_path=./eval/data # Path to the folder where the graph and config file will be copied and the results will be stored
prefix_dir=new_exp_dpsgd_gs_$(date '+%Y-%m-%dT%H:%M')
python_bin=./.venv/decentralizepy_env/bin
env_python=$python_bin/python
script_path=./tutorial/DPSGDwithNonIID # Path to the folder where the run_IDCAwPS.sh is located
eval_file=$decpy_path/testingDPSGDnIID.py # decentralized driver code (run on each machine)

machines=1 # number of machines in the runtime
tot_sample_seen=64000 # basically ideal_iter * ideal_rounds * ideal_B
test_after=20
log_level=INFO #INFO # DEBUG | INFO | WARN | CRITICAL
procs_per_machine=20

m=0 # machine id corresponding consistent with ip.json

echo "All started at $(date '+%Y-%m-%dT%H:%M')!"

config_file=config_CIFAR_grid_search_dpsgd.ini
config_file_path=$script_path/configs/$config_file

B=(8 16 32)
steps=(5 7 10)

cp $config_file_path $run_path

for rounds in ${steps[@]}
do
    for mini_batch in ${B[@]}
    do

        if [ $rounds -eq 8 ] ||  [ $steps -eq 5 ]; then
            continue

        $python_bin/crudini --set $run_path/$config_file TRAIN_PARAMS batch_size  $mini_batch
        $python_bin/crudini --set $run_path/$config_file TRAIN_PARAMS rounds $rounds
        # iterations=$(bc <<< "scale=0; $base_iterations * $rounds * $mini_batch / 128")
        iterations=$(($tot_sample_seen/$mini_batch/$rounds ))
        # iterations=$base_iterations

        echo iterations $iterations
        echo rounds $rounds
        echo mini_batch $mini_batch
        echo M is $m
        echo procs per machine is $procs_per_machine

        log_dir=$run_path/$prefix_dir/B_${mini_batch}_r_${rounds}_$(date '+%Y-%m-%dT%H:%M')/machine$m # in the eval folder
        echo $log_dir
        mkdir -p $log_dir
        start=$(date '+%s')
        $env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -ta $test_after -cf $run_path/$config_file -ll $log_level -wsd $log_dir
        end=$(date '+%s')
        duration=$((end-start))
        echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed."
    done
done
echo "All done at $(date '+%Y-%m-%dT%H:%M')!"