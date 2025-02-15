#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate jax_recreate

debug=0

if [ $debug -eq 1 ]; then
    proj_name=test
else
    proj_name=09_13_cql_widowx
fi

tpu_id=0
tpu_port=$(( $tpu_id+8820 ))
export PYTHONPATH=/home/asap7772/kun2/jaxrl2_finetuning_benchmark/:$PYTHONPATH; 
export PYTHONPATH=/home/asap7772/kun2/finetuning_benchmark/:$PYTHONPATH; 
export PYTHONPATH=/home/asap7772/kun2/finetuning_benchmark/data_collection/:$PYTHONPATH;
export EXP=/home/asap7772/kun2/jaxrl2_finetuning_benchmark/experiment_output
export DATA=/nfs/nfs1/

seed=1
cql_alpha=5
dry_run=0

total_runs=0
max_runs=32
gpu_id=0
which_devices=(0 1 2 3 4 5 6 7)


alphas=(0.1 1 5 10)
if [ $debug -eq 1 ]; then
    max_runs=1
    datasets=(debug)
else
    datasets=(sorting_nobinnoise sorting_nonzerobinnoise sorting sorting_pickplace)
fi

for alpha in ${alphas[@]}; do
for dataset in ${datasets[@]}; do
for calql in 0; do

prefix=${proj_name}_${dataset}_cql_alpha_${alpha}_dataset_${dataset}_seed_${seed}
which_gpu=${which_devices[$gpu_id % ${#which_devices[@]}]}
export CUDA_VISIBLE_DEVICES=$which_gpu
echo "Running on GPU $which_gpu"

export CUDA_VISIBLE_DEVICES=$which_gpu
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$which_gpu


command="XLA_PYTHON_CLIENT_PREALLOCATE=false python3 examples/launch_train_widowx_cql.py \
--prefix $prefix \
--cql_alpha $alpha \
--wandb_project ${proj_name} \
--batch_size 64 \
--encoder_type impala  \
--policy_encoder_type impala \
--actor_lr 0.0001 \
--critic_lr 0.0003 \
--dataset $dataset \
--seed $seed \
--offline_finetuning_start -1 \
--online_start 10000000000000 \
--max_steps  10000000000000 \
--eval_interval 1000 \
--log_interval 1000 \
--eval_episodes 20 \
--tpu_port $tpu_port \
--bound_q_with_mc $calql \
--discount 0.99 \
--checkpoint_interval 10000000000000 \
--bc_hotstart 50000 \
--multi_grad_step 5"

echo $command

if [ $dry_run -eq 0 ]; then
    if [ $debug -eq 1 ]; then
        eval $command
    else
        eval $command &
        sleep 10
    fi
fi

gpu_id=$(( $gpu_id+1 ))
if [ $gpu_id -eq 7 ]; then
    gpu_id=0
fi

total_runs=$(( $total_runs+1 ))
if [ $total_runs -eq $max_runs ]; then
    exit
fi

done
done
done