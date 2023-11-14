#!/bin/bash
#SBATCH --partition=svl
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --mem=128G
#SBATCH --job-name="dinstand"
#SBATCH --account=viscam

cd /iris/u/khatch/vd5rl/jaxrl2-irisfork/examples
source ~/.bashrc
# conda init bash
source /iris/u/khatch/anaconda3/bin/activate
# source activate jaxrlfork
conda activate jaxrlfork

unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL="egl"
export KITCHEN_DATASETS=/iris/u/khatch/vd5rl/datasets/diversekitchen
export STANDARD_KITCHEN_DATASETS=/iris/u/khatch/preliminary_experiments/model_based_offline_online/LOMPO/data/kitchen2/kitchen_demos_multitask_lexa_view_and_wrist_npz
export RELAY_POLICY_REPO="/iris/u/khatch/vd5rl/finetuning_benchmark/benchmark/domains/relay-policy-learning/adept_envs"

echo $SLURM_JOB_GPUS
export GPUS=$SLURM_JOB_GPUS
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.7

which python
which python3
nvidia-smi
pwd
ls -l /usr/local
python3 -u gpu_test.py


XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u train_offline_pixels_kitchen.py \
--task "standardkitchen_indistribution" \
--tqdm=true \
--project bench_standardkitchen_debug3 \
--algorithm ddpm_bc \
--proprio=true \
--description proprio \
--eval_episodes 50 \
--eval_interval 10_000 \
--online_eval_interval 10_000 \
--log_interval 1000 \
--max_gradient_steps 500_000 \
--max_online_gradient_steps 500_000 \
--finetune_online=false \
--replay_buffer_size 700_000 \
--batch_size 256 \
--im_size 64 \
--use_wrist_cam=false \
--camera_ids "12" \
--seed 0 