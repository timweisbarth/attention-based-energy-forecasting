#!/bin/bash
#SBATCH --job-name="ftS_transformerHP0_2_3_1"
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

for pred_len in 24; do
    for learning_rate in 0.001 0.005 0.01; do
        for batch_size in 32 64 128; do 
            for hpos in "2 1 32 4"; do
                read e_layers d_layers d_model n_heads <<< $hpos
                srun python3 -u run.py \
                  --des $current_folder \
                  --checkpoints ./checkpoints/$current_folder \
                  --is_training 1 \
                  --root_path ./data/preproc/ \
                  --data_path smard_data_DE.csv \
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
                  --factor 3 \
                  --enc_in 1 \
                  --dec_in 1 \
                  --c_out 1 \
                  --target "load" \
                  --itr 1 \
                  --learning_rate $learning_rate \
                  --train_epochs 30 \
                  --batch_size $batch_size \
                  --patience 3
            done  
        done
    done
done



