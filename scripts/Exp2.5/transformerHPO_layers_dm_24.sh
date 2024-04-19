#!/bin/bash
#SBATCH --job-name="ftS_transformerHP02_2_5"
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-galvani
#SBATCH --time 1-20:00:00 #
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x-%j.out  # cannot use $WORK 
#SBATCH --error=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x_%j.err

# useful for debugging
scontrol show job $SLURM_JOB_ID
nvidia-smi # only if you requested any gpus

current_folder=$(echo "${0}" | awk -F'/' '{for(i=1; i<=NF; i++) if($i ~ /^Exp/) print $i}')

for pred_len in 24 336
do
    for hpos in "4 3 32 4 0.001 64" "6 6 32 4 0.001 64" \
                "3 2 64 8 0.001 64" "4 3 64 8 0.001 64" "6 6 64 8 0.001 64" \
                "2 1 128 8 0.001 64" "3 2 128 8 0.001 64" "4 3 128 8 0.001 64" "6 6 128 8 0.001 64" \
                "2 1 256 8 0.0005 32" "3 2 256 8 0.0005 32" "4 3 256 8 0.0005 32" "6 6 256 8 0.0005 32" \
                "2 1 512 8 0.0005 32" "3 2 512 8 0.0005 32" "4 3 512 8 0.0005 32"; do
    
            read e_layers d_layers d_model n_heads lr bs <<< $hpos
            srun python3 -u run.py \
              --is_training 1 \
              --des $current_folder \
              --checkpoints ./checkpoints/$current_folder \
              --root_path ./data/preproc/ \
              --data_path smard_data.csv \
              --model_id 'load' \
              --model Transformer \
              --data smard \
              --features S \
              --seq_len 96 \
              --label_len 48 \
              --pred_len $pred_len \
              --e_layers $e_layers \
              --d_layers $d_layers \
              --d_model $d_model \
              --d_ff $(($d_model * 4)) \
              --n_heads $n_heads \
              --learning_rate $lr \
              --batch_size $bs \
              --factor 3 \
              --enc_in 1 \
              --dec_in 1 \
              --c_out 1 \
              --target "load" \
              --itr 1 \

        done
    done
done