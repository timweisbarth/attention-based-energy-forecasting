#!/bin/bash
#SBATCH --job-name="ftM_LSTM_weather_default_3_5"
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
current_folder="Exp3.5"

for d_model in "256" "512" "1024"; do
    for hpo in "0.05 0.01" "0.2 0.1"; do
        for lradj in "TST" "type1"; do
            for lr in "0.0005" "0.0001"; do
                read dropout weight_decay <<< $hpo
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
                --pred_len 192 \
                --e_layers 2 \
                --d_model $d_model \
                --learning_rate $lr \
                --batch_size 64 \
                --enc_in 15 \
                --dec_in 15 \
                --c_out 15 \
                --target "load_DE" \
                --itr 1 \
                --train_epochs 15 \
                --patience 6 \
                --optim adamW \
                --lradj $lradj \
                --weight_decay $weight_decay \
                --dropout $dropout \
                --pct_start 0.05 \
            
            done
        done
    done
done


