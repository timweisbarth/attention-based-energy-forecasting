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
for hpo in "24 128 3" "96 128 3" "192 256 6" "336 256 6" "720 256 6"; do
    read pred_len d_model layers <<< $hpo
    srun python3 -u run.py \
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
    --batch_size 64 \
    --enc_in 15 \
    --dec_in 15 \
    --c_out 15 \
    --target "load" \
    --itr 1 \
    --train_epochs 30 \
    --patience 6 \

done

# iTransformer
for hpo in "24 256" "96 256" "192 256" "336 256" "720 512"; do
    read pred_len d_model <<< $hpo
    srun python3 -u run.py \
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
    --batch_size 64 \
    --enc_in 15 \
    --dec_in 15 \
    --c_out 15 \
    --target "load" \
    --itr 1 \
    --train_epochs 30 \
    --patience 6 \
    
done

# LSTM
for hpo in "24 512 2" "96 512 2" "192 512 2" "336 512 2" "720 512 2"; do
    read pred_len d_model layers <<< $hpo
    srun python3 -u run.py \
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
    --batch_size 64 \
    --enc_in 15 \
    --dec_in 15 \
    --c_out 15 \
    --target "load" \
    --itr 1 \
    --train_epochs 30 \
    --patience 6 \

done


