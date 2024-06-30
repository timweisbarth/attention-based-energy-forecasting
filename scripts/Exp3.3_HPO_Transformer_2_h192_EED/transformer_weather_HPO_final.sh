#!/bin/bash
#SBATCH --job-name="ftM_transformer_HPO_weather_3_3"
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-galvani
#SBATCH --time 1-00:00:00 #
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x-%j.out  # cannot use $WORK 
#SBATCH --error=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x_%j.err

# useful for debugging
scontrol show job $SLURM_JOB_ID
nvidia-smi # only if you requested any gpus

#current_folder=$(echo "${0}" | awk -F'/' '{for(i=1; i<=NF; i++) if($i ~ /^Exp/) print $i}')
current_folder="Exp3.3"


for hpo in "64 6" "128 6" "256 3" "64 3"; do
    for lr in "0.0005" "0.0001"; do
        read d_model layers <<< $hpo
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
        --pred_len 192 \
        --e_layers $layers \
        --d_layers $layers \
        --d_model $d_model \
        --d_ff $(($d_model * 4)) \
        --n_heads 8 \
        --learning_rate $lr \
        --batch_size 64 \
        --enc_in 15 \
        --dec_in 15 \
        --c_out 15 \
        --target "load_DE" \
        --itr 1 \
        --train_epochs 13 \
        --patience 6 \
        --optim adamW \
        --lradj type1 \
        --dropout 0.05 \
        --weight_decay 0.01 \

    done
done


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
    --pred_len 192 \
    --e_layers 6 \
    --d_layers 6 \
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
    --train_epochs 13 \
    --patience 6 \
    --optim adamW \
    --lradj type1 \
    --dropout 0.5 \
    --weight_decay 0.05 \

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
    --pred_len 192 \
    --e_layers 6 \
    --d_layers 6 \
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
    --train_epochs 13 \
    --patience 6 \
    --optim adam \
    --lradj type1 \
    --dropout 0.5 \
    --weight_decay 0.0001 \


#DoubleDescent? ca3h
#srun python3 -u run.py \
#    --is_training 1 \
#    --des $current_folder \
#    --checkpoints ./checkpoints/$current_folder \
#    --root_path ./data/preproc/ \
#    --data_path smard_plus_weather_without_LUandAT.csv \
#    --model_id 'multi' \
#    --model Transformer \
#    --data smard_w_weather \
#    --including_weather 1 \
#    --features M \
#    --seq_len 336 \
#    --label_len 168 \
#    --pred_len 192 \
#    --e_layers 6 \
#    --d_layers 6 \
#    --d_model 256 \
#    --d_ff 1024 \
#    --n_heads 8 \
#    --learning_rate 0.0001 \
#    --batch_size 64 \
#    --enc_in 15 \
#    --dec_in 15 \
#    --c_out 15 \
#    --target "load_DE" \
#    --itr 1 \
#    --train_epochs 30 \
#    --patience 30 \
#    --optim adamW \
#    --lradj TST \
#    --dropout 0.05 \
#    --weight_decay 0.1 \
#    --pct_start 0.05 \

