#!/bin/bash

#nfs
nfs_decpy=/mnt/nfs/thiba/decentralizepy
eval_file=$nfs_decpy/eval/testingDPSGDnIID.py # decentralized driver code (run on each machine)
script_path=$nfs_decpy/tutorial/DPSGDwithNonIID # Path to the folder where the run_IDCAwPS.sh is located

config_file=config_IMGNETTE_grid_search_dpsgd.ini
config_file_path=$script_path/configs/$config_file

# m=0 # machine id corresponding consistent with ip.json
ip_machines=$nfs_decpy/tutorial/DPSGDwithNonIID/ip_nfs.json # !!!
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

machines=4 # number of machines in the runtime  # !!!
test_after=40
log_level=DEBUG #INFO # DEBUG | INFO | WARN | CRITICAL
procs_per_machine=5 # =32/4
iterations=801


prefix_dir=imgnette_new_exp_dpsgd_gs_param_$(date '+%Y-%m-%dT%H:%M')

echo "All started at $(date '+%Y-%m-%dT%H:%M')!"

#first download dataset
$env_python $machine_decpy/download_imagenette.py

setups=(1 2 3)

cp $config_file_path $run_path

for setup in ${setups[@]}
do
    case $setup in
        1)
            rounds=10
            mini_batch=32
            ;;
        2)
            rounds=5
            mini_batch=8
            ;;
        3)
            rounds=5
            mini_batch=32
            ;;
    esac

    $python_bin/crudini --set $run_path/$config_file TRAIN_PARAMS batch_size  $mini_batch
    $python_bin/crudini --set $run_path/$config_file TRAIN_PARAMS rounds $rounds

    $python_bin/crudini --set $run_path/$config_file COMMUNICATION addresses_filepath $ip_machines

    echo iterations $iterations
    echo procs per machine is $procs_per_machine
    echo rounds $rounds
    echo mini_batch $mini_batch

    log_dir_from_decpy=eval/data/$prefix_dir/B_${mini_batch}_r_${rounds}_$(date '+%Y-%m-%dT%H:%M')
    log_dir=$machine_decpy/$log_dir_from_decpy/machine$m # in the eval folder
    echo $log_dir
    mkdir -p $log_dir
    start=$(date '+%s')

    $env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -ta $test_after -cf $run_path/$config_file -ll $log_level -wsd $log_dir
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