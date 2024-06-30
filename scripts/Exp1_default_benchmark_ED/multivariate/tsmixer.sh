#!/bin/bash
#SBATCH --job-name="ftM_TSMixer_1"
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-galvani
#SBATCH --time 1-12:00:00 #
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x-%j.out  # cannot use $WORK 
#SBATCH --error=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x_%j.err

# useful for debugging
scontrol show job $SLURM_JOB_ID
nvidia-smi # only if you requested any gpus

for pred_len in 24 96 192 336 720
do
  srun python3 -u run.py \
    --is_training 1 \
    --root_path ./data/preproc/ \
    --data_path smard_data.csv \
    --model_id '' \
    --model TSMixer \
    --data smard \
    --features M \
    --seq_len 336 \
    --pred_len $pred_len \
    --e_layers 4 \
    --enc_in 3 \
    --c_out 3 \
    --target "load" \
    --itr 3 \
    --d_model 256 \
    --dropout 0.5 \
    --train_epochs 100 \
    --patience 10 \
    --batch_size 32 \
    --learning_rate 0.0001 \

done