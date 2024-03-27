#!/bin/bash
#SBATCH --job-name="ftS_patchtst_wind_2"
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-galvani
#SBATCH --time 1-15:00:00 #
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
      --model_id 'wind_gen' \
      --model PatchTST \
      --data smard \
      --features S \
      --seq_len 336 \
      --pred_len $pred_len \
      --enc_in 1 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --c_out 1 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --train_epochs 100 \
      --patience 10 \
      --lradj 'TST' \
      --pct_start 0.2 \
      --batch_size 32 \
      --learning_rate 0.0001 \
      --target "wind_gen" \
      --itr 3 \

done