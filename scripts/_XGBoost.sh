#!/bin/bash
#SBATCH --job-name="single_v100_job_for_xgb_test"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100
#SBATCH --time 0-10:00:00 # set maximum allowed runtime to 20h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/qb/work/ludwig/lqb853/ftS_xgb_%x-%j.out  # note one cannot use env variables like $WORK in #SBATCH statements

# useful for debugging
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested any gpus

ls $WORK # not necessary just here to illustrate that $WORK is avaible here

#export CUDA_VISIBLE_DEVICES=1
srun python3 -u run_non_deepl.py \
