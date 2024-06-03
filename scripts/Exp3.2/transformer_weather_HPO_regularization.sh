#!/bin/bash
#SBATCH --job-name="ftM_transformer_HPO_weather_3_2"
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
current_folder="Exp3.2"


#TODO: Choose correct optim!
#Regularization ca 6h
# Combination of 0.05 and 0.01 is already tested in other shell script
for hpo in "0.05 0.1" "0.2 0.01" "0.2 0.1";do
    for lradj in "type1" "TST";do
        read dropout weight_decay <<< $hpo
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
        --lradj $lradj \
        --dropout $dropout \
        --weight_decay $weight_decay \

    done
done

#Heads ca 1h
for heads in 16;do
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
    --train_epochs 15 \
    --patience 6 \
    --optim adamW \
    --lradj type1 \
    --n_heads $heads \

done

#DoubleDescent? ca4h
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
    --train_epochs 50 \
    --patience 50 \
    --optim adamW \
    --lradj TST \

