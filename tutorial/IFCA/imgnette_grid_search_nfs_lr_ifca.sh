#!/bin/bash

#nfs
nfs_decpy=/mnt/nfs/thiba/decentralizepy
eval_file=$nfs_decpy/eval/testingIFCA_restarts.py # decentralized driver code (run on each machine)
script_path=$nfs_decpy/tutorial/IFCA # Path to the folder where the run_IDCAwPS.sh is located

config_file=config_IMGNETTE_grid_search_ifca.ini

config_file_path=$script_path/configs/$config_file

# m=0 # machine id corresponding consistent with ip.json
ip_machines=$nfs_decpy/tutorial/IFCA/ip_nfs_B.json # !!!
m=$(cat $ip_machines | grep $HOSTNAME | awk '{print $1}' | cut -d'"' -f2)
echo M is $m


#machine
machine_decpy=/home/thiba/decentralizepy
# RUN it from the root folder of the project
python_bin=$machine_decpy/.venv/decentralizepy_env/bin
env_python=$python_bin/python

#copy stuff
cp $nfs_decpy/tutorial/download_imagenette.py $machine_decpy/
# cp $nfs_decpy/setup.py $machine_decpy/
# cp $nfs_decpy/setup.cfg $machine_decpy/
cp -r $nfs_decpy/src $machine_decpy/

#activate the virtual environment and PIP INSTSLL !! to update the decpy pacakge
source $python_bin/activate
# pip install .

run_path=$machine_decpy/eval/data # Path to the folder where the graph and config file will be copied and the results will be stored

cp $config_file_path $run_path

machines=3 # number of machines in the runtime  # !!!
test_after=80
log_level=DEBUG #INFO # DEBUG | INFO | WARN | CRITICAL
procs_per_machine=8 # =24/3
iterations=801


prefix_dir=imgnette_new_exp_ifca_gs_$(date '+%Y-%m-%dT%H:%M')

echo "All started at $(date '+%Y-%m-%dT%H:%M')!"

#first download dataset
$env_python $machine_decpy/download_imagenette.py

# lrs=(0.0005 0.001 0.005)


# for lr in ${lrs[@]}
# do
#     $python_bin/crudini --set $run_path/$config_file OPTIMIZER_PARAMS lr $lr
rounds=(5)


for round in ${rounds[@]}
do
    $python_bin/crudini --set $run_path/$config_file TRAIN_PARAMS rounds $round
    $python_bin/crudini --set $run_path/$config_file COMMUNICATION addresses_filepath $ip_machines

    echo iterations $iterations
    echo procs per machine is $procs_per_machine
    # echo lr is $lr
    echo round is $round

    log_dir_from_decpy=eval/data/$prefix_dir/lr_${lr}_$(date '+%Y-%m-%dT%H:%M')
    log_dir=$machine_decpy/$log_dir_from_decpy/machine$m # in the eval folder
    echo $log_dir
    mkdir -p $log_dir
    start=$(date '+%s')

    server_machine=0
    server_rank=-1
    working_rate=1.0

    $env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -wsd $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -ta $test_after -cf $run_path/$config_file -ll $log_level -sm $server_machine -sr $server_rank -wr $working_rate
    # touch $log_dir/started
    # sleep 2

    end=$(date '+%s')
    duration=$((end-start))
    echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed."

    #copy back the files
    mkdir -p $nfs_decpy/$log_dir_from_decpy
    cp -r $log_dir/ $nfs_decpy/$log_dir_from_decpy
done
echo "All done at $(date '+%Y-%m-%dT%H:%M')!"