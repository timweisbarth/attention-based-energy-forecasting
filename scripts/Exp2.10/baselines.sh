#!/bin/bash
#SBATCH --job-name="ftS_baselines_HPO_2_10"
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

#current_folder=$(echo "${0}" | awk -F'/' '{for(i=1; i<=NF; i++) if($i ~ /^Exp/) print $i}')
current_folder="Exp2.10"

# Include DLinear!
#Needs modification! Check LSTM and TSMixer to be consistent with prior experiments
for pred_len in 192; do
    for hpos in "1 512 96 0.0001" "2 512 96 0.0001" "4 512 96 0.0001" "2 128 96 0.0001" "2 256 96 0.0001" "2 512 96 0.0001" "2 512 96 0.001" "2 512 96 0.0005" "2 512 96 0.0001" "2 512 96 0.0001" "2 512 336 0.0001"
                    #read e_layers d_layers <<< $layer
                    read layers d_model seq_len lr <<< $hpos
                    srun python3 -u run.py \
                      --is_training 1 \
                      --des $current_folder \
                      --checkpoints ./checkpoints/$current_folder \
                      --root_path ./data/preproc/ \
                      --data_path smard_data_DE.csv \
                      --model_id 'load' \
                      --model LSTM \
                      --data smard \
                      --features S \
                      --seq_len $seq_len \
                      --pred_len $pred_len \
                      --e_layers $layers \
                      --d_model $d_model \
                      --learning_rate $lr \
                      --batch_size 32 \
                      --enc_in 1 \
                      --dec_in 1 \
                      --c_out 1 \
                      --target "load" \
                      --itr 1 \
                      --train_epochs 30 \
                      --patience 3
                done      
            done
        done
    done
done

for pred_len in 192; do
    for layers in 2 4; do
        for dim in 256 512; do
            for seq_len in 96 336; do
                for lr in 0.001 0.0005 0.0001; do
                    #read e_layers d_layers <<< $layer
                    read d_model n_heads <<< $dim
                    srun python3 -u run.py \
                      --is_training 1 \
                      --des $current_folder \
                      --checkpoints ./checkpoints/$current_folder \
                      --root_path ./data/preproc/ \
                      --data_path smard_data_DE.csv \
                      --model_id 'load' \
                      --model TSMixer \
                      --data smard \
                      --features S \
                      --seq_len $seq_len \
                      --pred_len $pred_len \
                      --e_layers $layers \
                      --d_model $d_model \
                      --learning_rate $lr \
                      --batch_size 32 \
                      --enc_in 1 \
                      --dec_in 1 \
                      --c_out 1 \
                      --target "load" \
                      --itr 1 \
                      --train_epochs 30 \
                      --patience 3
                done      
            done
        done
    done
done