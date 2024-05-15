#!/bin/bash
#SBATCH --job-name="ftM_benchmark_weather_default_3_1"
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-galvani
#SBATCH --time 1-22:00:00 #
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x-%j.out  # cannot use $WORK 
#SBATCH --error=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x_%j.err

# useful for debugging
scontrol show job $SLURM_JOB_ID
nvidia-smi # only if you requested any gpus

#current_folder=$(echo "${0}" | awk -F'/' '{for(i=1; i<=NF; i++) if($i ~ /^Exp/) print $i}')
current_folder="Exp3.1"

# Not finished! TODO
for pred_len in 24 48 192 336 720; do
    for hpo in "iTransformer 3" "Transformer" "TSMixer" "LSTM" "ModernTransformer"; do
                      read model layers <<< $hpo
                    
                      --is_training 1 \
                      --des $current_folder \
                      --checkpoints ./checkpoints/$current_folder \
                      --root_path ./data/preproc/ \
                      --data_path smard_data_DE.csv \
                      --model_id 'load' \
                      --model Transformer \
                      --data smard \
                      --features S \
                      --seq_len $seq_len \
                      --label_len $(expr $seq_len / 2) \
                      --pred_len $pred_len \
                      --e_layers $e_layers \
                      --d_layers $d_layers \
                      --d_model $d_model \
                      --d_ff $(($d_model * 4)) \
                      --n_heads $n_heads \
                      --learning_rate $lr \
                      --batch_size 32 \
                      --factor 3 \
                      --enc_in 1 \
                      --dec_in 1 \
                      --c_out 1 \
                      --target "load" \
                      --itr 1 \
                      --train_epochs 30 \
                      --patience 3

    done
done