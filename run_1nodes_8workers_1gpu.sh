#!/bin/bash

#SBATCH -c 32
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

#SBATCH --job-name=hpo_freezing
#SBATCH --open-mode=append  # important for multiple processes to share a log file

#SBATCH --error=./logs/%j_%a_%N_log.err
#SBATCH --output=./logs/%j_%a_%N_log.out

#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --time=23:59:00  # in minutes, 60=1 hour, 1440=1 day, 10800=1 week
#SBATCH --exclude=dlcgpu28,dlcgpu35
#SBATCH --mail-type=BEGIN,END,FAIL

source .venv/bin/activate

if [ -z "$1" ]; then
    echo "Error: group_name argument is required"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: n_unfrozen_layers argument is required"
    exit 1
fi

group_name=$1
n_unfrozen_layers=$2

# --unbuffered: 
#       enables multiple processes to log output instantly but can slow I/O overall
srun --unbuffered \
    --ntasks 8 \
    --gpus-per-task 1 \
    --cpus-per-task 4 \
    python layer_freeze/simple_cnn_cifar10.py \
        --nodes 1 \
        --gpus_per_node 8 \
        --cpus_per_node 4 \
        --group_name $group_name \
        --n_unfrozen_layers $n_unfrozen_layers

# end of file
