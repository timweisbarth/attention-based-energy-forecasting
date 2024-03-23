#!/bin/bash
#SBATCH --job-name="single_a100_job_for_test"
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-galvani
#SBATCH --time 0-00:20:00 # set maximum allowed runtime to 10min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/qb/work/ludwig/lqb853/%x-%j.out  # note one cannot use env variables like $WORK in #SBATCH statements

# useful for debugging
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested any gpus

ls $WORK # not necessary just here to illustrate that $WORK is avaible here

# add possibly other setup code here, e.g.
# - copy singularity images or data to local storage like /scratch_local
# - loads virtual envs, like with anaconda
# - set environments variables
# - determine commandline arguments for `srun` calls


for pred_len in 60
do
    srun python3 -u run.py \
      --is_training 1 \
      --root_path ./data/preproc/ \
      --data_path smard_data.csv \
      --model_id 'load' \
      --model DLinear \
      --data smard \
      --features S \
      --seq_len 336 \
      --pred_len $pred_len \
      --enc_in 1 \
      --batch_size 32 \
      --learning_rate 0.005 \
      --target "load" \
      --itr 1 \
      --patience 10 \
      --d_model 48 \
      --train_epochs 2 \

done