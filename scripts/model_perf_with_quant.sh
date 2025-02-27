#!/bin/bash

#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=16G

#SBATCH --job-name=model_perf
#SBATCH --open-mode=append  # important for multiple processes to share a log file

#SBATCH --error=./slurm_logs/%j_%a_%N_log.err
#SBATCH --output=./slurm_logs/%j_%a_%N_log.out

#SBATCH --partition=mldlc2_gpu-l40s
#SBATCH --time=8:00:00  # in minutes, 60=1 hour, 1440=1 day, 10800=1 week
#SBATCH --mail-type=FAIL

cd /work/dlcsmall1/carstent-layer-freeze || exit 1

source .venv/bin/activate

# --unbuffered: 
#       enables multiple processes to log output instantly but can slow I/O overall

for model in resnet18 resnet34 resnet50 resnet101 resnet152; do
    for batch_size in 1 32 64 128; do
        srun --unbuffered \
        --ntasks 1 \
        --gpus-per-task 1 \
        --cpus-per-task 16 \
        python test.py \
            --base_model $model \
            --batch_size $batch_size \
            --quantize_frozen_layers
    done
done

for model in resnet18 resnet34 resnet50 resnet101 resnet152; do
    for batch_size in 1 32 64 128; do
        srun --unbuffered \
        --ntasks 1 \
        --gpus-per-task 1 \
        --cpus-per-task 16 \
        python test.py \
            --base_model $model \
            --batch_size $batch_size
    done
done

# end of file
