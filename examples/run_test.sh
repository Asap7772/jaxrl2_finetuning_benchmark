eval "$(conda shell.bash hook)"
conda activate jax_recreate

which_gpu=0
export PYTHONPATH=/home/asap7772/kun2/jaxrl2_finetuning_benchmark/:$PYTHONPATH; 
export PYTHONPATH=/home/asap7772/kun2/finetuning_benchmark/:$PYTHONPATH; 
export PYTHONPATH=/home/asap7772/kun2/finetuning_benchmark/data_collection/:$PYTHONPATH;
export EXP=/home/asap7772/kun2/jaxrl2_finetuning_benchmark/experiment_output
export DATA=/nfs/nfs1/

export CUDA_VISIBLE_DEVICES=$which_gpu
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$which_gpu

python jaxrl2/data/eps_transition_dataset.py