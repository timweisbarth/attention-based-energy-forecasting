#!/bin/bash
#SBATCH --job-name="ftM_TestSet_Transformer"
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-galvani
#SBATCH --time 0-15:00:00 #
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x-%j.out  # cannot use $WORK 
#SBATCH --error=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x_%j.err

# useful for debugging
scontrol show job $SLURM_JOB_ID
nvidia-smi # only if you requested any gpus

#current_folder=$(echo "${0}" | awk -F'/' '{for(i=1; i<=NF; i++) if($i ~ /^Exp/) print $i}')
current_folder="Exp4"

# MultiXL, Transformer
for hpos in "720 5"; do
    read pred_len layers <<< $hpos
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
    --e_layers $layers \
    --d_layers $layers \
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
    --final_run_train_on_train_and_val \

done

# Load, Transformer
for hpos in "24 128 3 32" "96 128 3 32" "192 256 6 64" "336 256 6 64" "720 256 5 64"; do
    read pred_len d_model layers bs <<< $hpos
    srun python3 -u run.py \
        --is_training 1 \
        --des $current_folder \
        --checkpoints ./checkpoints/$current_folder \
        --root_path ./data/preproc/ \
        --data_path smard_data_DE.csv \
        --model_id 'load' \
        --model Transformer \
        --data smard \
        --features S \
        --seq_len 336 \
        --label_len 168 \
        --pred_len $pred_len \
        --e_layers $layers \
        --d_layers $layers \
        --d_model $d_model \
        --d_ff $(($d_model * 4)) \
        --n_heads 8 \
        --learning_rate 0.0005 \
        --batch_size $bs \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --target "load" \
        --itr 1 \
        --train_epochs 30 \
        --patience 3 \
        --final_run_train_on_train_and_val \

done