#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate jax_recreate

debug=0
dry_run=0

if [ $debug -eq 1 ]; then
    proj_name=test
else
    proj_name=11_13_widowx_bc
fi

# proj_name=test
tpu_id=0
tpu_port=$(( $tpu_id+8820 ))
export PYTHONPATH=/home/asap7772/kun2/jaxrl2_finetuning_benchmark/:$PYTHONPATH; 
export PYTHONPATH=/home/asap7772/kun2/finetuning_benchmark/:$PYTHONPATH; 
export PYTHONPATH=/home/asap7772/kun2/finetuning_benchmark/data_collection/:$PYTHONPATH;
export EXP=/home/asap7772/kun2/jaxrl2_finetuning_benchmark/experiment_output
export DATA=/nfs/nfs1/

which_exp=${1:--1}
total_runs=0
max_runs=8

num_seeds=2
which_devices=(0 1 2 3 4 5 6 7)
actor_lrs=(0.0001)
datasets=(sorting_nobinnoise sorting_nonzerobinnoise sorting sorting_pickplace)

if [ $debug -eq 1 ]; then
    max_runs=1
    actor_lr=(0.01)
    datasets=(debug)
fi

for dataset in ${datasets[@]}; do
for actor_lr in ${actor_lrs[@]}; do
for seed in $(seq 1 $num_seeds); do

prefix=${proj_name}_${dataset}_dataset_${dataset}_seed_${seed}
which_gpu=${which_devices[$gpu_id % ${#which_devices[@]}]}

export CUDA_VISIBLE_DEVICES=$which_gpu
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$which_gpu

command="XLA_PYTHON_CLIENT_PREALLOCATE=false python3 examples/launch_train_widowx_bc.py \
--prefix $prefix \
--wandb_project ${proj_name} \
--batch_size 256 \
--encoder impala  \
--actor_lr $actor_lr \
--dataset $dataset \
--seed $seed \
--offline_finetuning_start -1 \
--online_start 10000000000000 \
--max_steps  10000000000000 \
--eval_interval 1000 \
--log_interval 1000 \
--eval_episodes 20 \
--checkpoint_interval 10000000000000 \
--tpu_port $tpu_port"

if [[ $which_exp -eq $total_runs ]]; then
    echo "Running on GPU $which_gpu"
    echo -e "$command\n"
    if [ $dry_run -eq 0 ]; then
        echo "Running experiment $which_exp"
        eval $command
    fi
elif [[ $which_exp -lt 0 ]]; then
    echo "Running on GPU $which_gpu"
    echo -e "$command\n"
    if [ $dry_run -eq 0 ]; then
        eval $command &
        sleep 100
    fi
fi

total_runs=$(( $total_runs+1 ))
if [ $total_runs -eq $max_runs ]; then
    echo "Total runs: $total_runs"
    exit
fi

done
done
done

echo "Total runs: $total_runs"