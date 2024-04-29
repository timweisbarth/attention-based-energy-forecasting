#!/bin/bash
#SBATCH --job-name="ftS_baselines_HPO_2_10"
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-galvani
#SBATCH --time 1-03:00:00 #
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x-%j.out  # cannot use $WORK 
#SBATCH --error=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x_%j.err

# useful for debugging
scontrol show job $SLURM_JOB_ID
nvidia-smi # only if you requested any gpus

#current_folder=$(echo "${0}" | awk -F'/' '{for(i=1; i<=NF; i++) if($i ~ /^Exp/) print $i}')
current_folder="Exp2.10"

for pred_len in 24 336; do
    for hpos in "2 512 96 0.0001" \
                "1 512 96 0.0001"  "4 512 96 0.0001" \
                "2 256 96 0.0001" "2 1024 96 0.0001" \
                "2 512 96 0.001" "2 512 96 0.0005" "2 512 96 0.00005" \
                "2 512 336 0.0001"; do
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

for pred_len in 24 336; do
    for hpos in "4 256 336 0.0001" \
                "2 256 336 0.0001" "6 256 336 0.0001" \
                "4 128 336 0.0001" "4 512 336 0.0001" \
                "4 256 336 0.001" "4 256 336 0.0005" "4 256 336 0.00005" \
                "4 256 96 0.0001"; do
        read layers d_model seq_len lr <<< $hpos
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

for pred_len in 24 336; do
    for hpos in "336 0.005" \
                "336 0.01" "336 0.001" "336 0.0005" "336 0.0001" \
                "96 0.005"; do
        read seq_len lr <<< $hpos
        srun python3 -u run.py \
          --is_training 1 \
          --des $current_folder \
          --checkpoints ./checkpoints/$current_folder \
          --root_path ./data/preproc/ \
          --data_path smard_data_DE.csv \
          --model_id 'load' \
          --model DLinear \
          --data smard \
          --features S \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --learning_rate $lr \
          --batch_size 32 \
          --enc_in 1 \
          --target "load" \
          --itr 1 \
          --train_epochs 30 \
          --patience 3
    done
done