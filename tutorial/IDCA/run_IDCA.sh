#!/bin/bash

# RUN it from the root folder of the project

decpy_path=./eval # Path to eval folder
graph=regular_16.txt # Absolute path of the graph file generated using the generate_graph.py script
run_path=./eval/data # Path to the folder where the graph and config file will be copied and the results will be stored
config_file=config.ini
script_path=./tutorial/IDCA # Path to the folder where the run_IDCA.sh is located
cp $script_path/$graph $script_path/$config_file $run_path

# env_python=~/miniconda3/envs/decpy/bin/python3 # Path to python executable of the environment | conda recommended
# env_python=~/.local/share/virtualenvs/decentralizepy-Ewz13z8M/bin/python
env_python=./.venv/decentralizepy_env/bin/python
machines=1 # number of machines in the runtime
iterations=80
test_after=2
eval_file=$decpy_path/testingIDCA.py # decentralized driver code (run on each machine)
log_level=INFO #INFO # DEBUG | INFO | WARN | CRITICAL

m=0 # machine id corresponding consistent with ip.json
echo M is $m

procs_per_machine=16 # 16 processes on 1 machine
echo procs per machine is $procs_per_machine

log_dir=$run_path/$(date '+%Y-%m-%dT%H:%M')/machine$m # in the eval folder
mkdir -p $log_dir

$env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $run_path/$graph -ta $test_after -cf $run_path/$config_file -ll $log_level -wsd $log_dir