#!/bin/bash

#nfs
nfs_decpy=/mnt/nfs/thiba/decentralizepy
eval_file=$nfs_decpy/eval/testingIDCAwPS.py # decentralized driver code (run on each machine)
script_path=$nfs_decpy/tutorial/IDCA # Path to the folder where the run_IDCAwPS.sh is located

config_file=config_FLICKR_idca.ini

config_file_path=$script_path/configs/$config_file

# m=0 # machine id corresponding consistent with ip.json
ip_machines=$nfs_decpy/tutorial/ip_files/ip_nfs_B.json
m=$(cat $ip_machines | grep $HOSTNAME | awk '{print $1}' | cut -d'"' -f2)
echo M is $m


#machine
machine_decpy=/home/thiba/decentralizepy
# RUN it from the root folder of the project
python_bin=$machine_decpy/.venv/decentralizepy_env/bin
env_python=$python_bin/python

#copy stuff
cp $nfs_decpy/download_dataset.py $machine_decpy/
# cp $nfs_decpy/setup.py $machine_decpy/
# cp $nfs_decpy/setup.cfg $machine_decpy/
cp -r $nfs_decpy/src $machine_decpy/

#activate the virtual environment and PIP INSTSLL !! to update the decpy pacakge
source $python_bin/activate
# pip install .

run_path=$machine_decpy/eval/data # Path to the folder where the graph and config file will be copied and the results will be stored

cp $config_file_path $run_path

machines=4 # number of machines in the runtime
test_after=40
log_level=INFO #INFO # DEBUG | INFO | WARN | CRITICAL
procs_per_machine=4 # =32/4
iterations=400


prefix_dir=flickr_idca_gs_share_4800_v2_$(date '+%Y-%m-%dT%H:%M')

echo "All started at $(date '+%Y-%m-%dT%H:%M')!"

#first download dataset
$env_python $machine_decpy/download_dataset.py

lrs=(0.1) # 0.0001 is too small, 0.3 prob too big


for lr in ${lrs[@]}
do
    $python_bin/crudini --set $run_path/$config_file OPTIMIZER_PARAMS lr $lr
    $python_bin/crudini --set $run_path/$config_file COMMUNICATION addresses_filepath $ip_machines

    echo iterations $iterations
    echo procs per machine is $procs_per_machine
    echo lr $lr

    log_dir_from_decpy=eval/data/$prefix_dir/lr_${lr}_$(date '+%Y-%m-%dT%H:%M')
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