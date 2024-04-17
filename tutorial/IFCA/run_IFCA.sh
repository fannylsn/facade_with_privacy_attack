#!/bin/bash

# RUN it from the root folder of the project

decpy_path=./eval # Path to eval folder
run_path=./eval/data # Path to the folder where the config file will be copied and the results will be stored
config_file=configCIFAR.ini
script_path=./tutorial/IFCA # Path to the folder where the run_IFCA.sh is located
cp $script_path/$config_file $run_path

env_python=./.venv/decentralizepy_env/bin/python
machines=1 # number of machines in the runtime
iterations=80
test_after=4
eval_file=$decpy_path/testingIFCA.py # decentralized driver code (run on each machine)
log_level=INFO #INFO # DEBUG | INFO | WARN | CRITICAL

server_rank=-1
server_machine=0
working_rate=1

m=0 # machine id corresponding consistent with ip.json
echo M is $m

procs_per_machine=18 # 16 processes on 1 machine
echo procs per machine is $procs_per_machine

log_dir=$run_path/$(date '+%Y-%m-%dT%H:%M')_ifca/machine$m # in the eval folder
mkdir -p $log_dir

$env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -wsd $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations  -ta $test_after -cf $run_path/$config_file -ll $log_level -sm $server_machine -sr $server_rank -wr $working_rate