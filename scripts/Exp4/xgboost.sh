#!/bin/bash
#SBATCH --job-name="ftS_and_M_xgboost_1"
#SBATCH --gres=gpu:1
#SBATCH --partition=2080-galvani
#SBATCH --time 1-20:00:00 #
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x-%j.out  # cannot use $WORK 
#SBATCH --error=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x_%j.err

# useful for debugging
scontrol show job $SLURM_JOB_ID
nvidia-smi # only if you requested any gpus

# WARNING: It is necessary to specify the specs of the model in main() of the file run_non_deepl.py
srun python3 -u run_non_deepl.py \
