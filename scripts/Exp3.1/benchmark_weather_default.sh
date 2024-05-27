#!/bin/bash
#SBATCH --job-name="ftM_benchmark_weather_default_3_1"
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
current_folder="Exp3.1"

#Transformer
for hpo in "24 128 3 32"  "96 128 3 32" "192 256 6 64" "336 256 6 64" "720 256 6 64"; do
    read pred_len d_model layers bs <<< $hpo

    --is_training 1 \
    --des $current_folder \
    --checkpoints ./checkpoints/$current_folder \
    --root_path ./data/preproc/ \
    --data_path smard_plus_weather_without_LUandAT.csv \
    --model_id 'load' \
    --model Transformer \
    --data smard_w_weather \
    --including_weather 1 \
    --features M \
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
    --enc_in 15 \
    --dec_in 15 \
    --c_out 15 \
    --target "load" \
    --itr 1 \
    --train_epochs 30 \
    --patience 5 \

done

# iTransformer
for hpo in "24 256"  "96 256" "192 256" "336 256" "720 512"; do
    read pred_len d_model <<< $hpo

    --is_training 1 \
    --des $current_folder \
    --checkpoints ./checkpoints/$current_folder \
    --root_path ./data/preproc/ \
    --data_path smard_plus_weather_without_LUandAT.csv \
    --model_id 'load' \
    --model iTransformer \
    --data smard_w_weather \
    --including_weather 1 \
    --features M \
    --seq_len 336 \
    --label_len 168 \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_model $d_model \
    --d_ff $d_model \
    --n_heads 8 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --enc_in 15 \
    --dec_in 15 \
    --c_out 15 \
    --target "load" \
    --itr 1 \
    --train_epochs 30 \
    --patience 5 \
    
done

# LSTM
for hpo in "24 512 1"  "96 512 1" "192 1024 2" "336 1024 2" "720 1024 2"; do
    read pred_len d_model layers <<< $hpo

    --is_training 1 \
    --des $current_folder \
    --checkpoints ./checkpoints/$current_folder \
    --root_path ./data/preproc/ \
    --data_path smard_plus_weather_without_LUandAT.csv \
    --model_id 'load' \
    --model LSTM \
    --data smard_w_weather \
    --including_weather 1 \
    --features M \
    --seq_len 96 \
    --pred_len $pred_len \
    --e_layers $layers \
    --d_model $d_model \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --enc_in 15 \
    --dec_in 15 \
    --c_out 15 \
    --target "load" \
    --itr 1 \
    --train_epochs 30 \
    --patience 5 \

done


