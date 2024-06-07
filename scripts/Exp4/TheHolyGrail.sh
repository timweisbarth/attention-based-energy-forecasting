#!/bin/bash
#SBATCH --job-name="ftM_TestSet_2080"
#SBATCH --gres=gpu:1
#SBATCH --partition=2080-galvani
#SBATCH --time 1-00:00:00 #
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x-%j.out  # cannot use $WORK 
#SBATCH --error=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x_%j.err

# useful for debugging
scontrol show job $SLURM_JOB_ID
nvidia-smi # only if you requested any gpus

#current_folder=$(echo "${0}" | awk -F'/' '{for(i=1; i<=NF; i++) if($i ~ /^Exp/) print $i}')
current_folder="Exp4"

# All models except Transformer on the test set due to memory of 2080
# Divide in load, solar? and MultiXL
# Integrate test_set in run.py and in data loader

# MultiXL, Transformer
for pred_len in 24 96 192 336 720; do
    srun python3 -u run.py \
    --is_training 1 \
    --des $current_folder \
    --checkpoints ./checkpoints/$current_folder \
    --root_path ./data/preproc/ \
    --data_path smard_plus_weather_without_LUandAT.csv \
    --model_id 'ftM' \
    --model Transformer \
    --data smard_w_weather \
    --including_weather 1 \
    --features M \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 3 \
    --d_model 256 \
    --d_ff 1024 \
    --n_heads 8 \
    --learning_rate 0.0005 \
    --batch_size 64 \
    --enc_in 15 \
    --dec_in 15 \
    --c_out 15 \
    --target "load_DE" \
    --itr 1 \
    --train_epochs 30 \
    --patience 3 \
    --optim adamW \
    --lradj type1 \
    --dropout 0.05 \
    --weight_decay 0.01 \
    --test_set 1 \

done

# Multi XL, LSTM
for pred_len in 24 96 192 336 720; do
    srun python3 -u run.py \
    --is_training 1 \
    --des $current_folder \
    --checkpoints ./checkpoints/$current_folder \
    --root_path ./data/preproc/ \
    --data_path smard_plus_weather_without_LUandAT.csv \
    --model_id 'ftM' \
    --model LSTM \
    --data smard_w_weather \
    --including_weather 1 \
    --features M \
    --seq_len 96 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_model  1024 \
    --learning_rate 0.0005 \
    --batch_size 64 \
    --enc_in 15 \
    --dec_in 15 \
    --c_out 15 \
    --target "load_DE" \
    --itr 1 \
    --train_epochs 30 \
    --patience 3 \
    --optim adamW \
    --lradj "OneCycle" \
    --weight_decay 0.01 \
    --dropout 0.05 \
    --pct_start 0.05 \
    --test_set 1 \

done
            







