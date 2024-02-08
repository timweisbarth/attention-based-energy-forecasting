#!/bin/bash
#SBATCH --job-name="single_v100_job_for_lstm_test"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100
#SBATCH --time 0-00:10:00 # set maximum allowed runtime to 10min
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


export CUDA_VISIBLE_DEVICES=0

srun python3 -u run.py \
  --is_training 1 \
  --root_path ./data/preproc/ \
  --data_path smard_data.csv \
  --model_id 'load' \
  --model LSTM \
  --data smard \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 96 \
  --d_ff 192 \
  --target "load" \
  --itr 1 \
  --train_epochs 15\