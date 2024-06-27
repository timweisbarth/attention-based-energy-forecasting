#!/bin/bash
#SBATCH --job-name="ftS_patchtst_HPO_192"
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-galvani
#SBATCH --time 1-23:00:00 #
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x-%j.out  # cannot use $WORK 
#SBATCH --error=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x_%j.err

# useful for debugging
scontrol show job $SLURM_JOB_ID
nvidia-smi # only if you requested any gpus

current_folder="Exp2.11"

# lr: larger because training took pretty long
for pred_len in 192; do
    for layers in "3" "6"; do
        for hpo in "32 4" "64 8" "128 16" "256 16" "512 16"; do
            for seq_len in 336 512; do
                for lr in 0.0001; do
                    read d_model n_heads <<< $hpo
                    srun python3 -u run.py \
                      --is_training 1 \
                      --des $current_folder \
                      --checkpoints ./checkpoints/$current_folder \
                      --root_path ./data/preproc/ \
                      --data_path smard_data_DE.csv \
                      --model_id 'load' \
                      --model PatchTST \
                      --data smard \
                      --features S \
                      --seq_len $seq_len \
                      --pred_len $pred_len \
                      --e_layers $layers \
                      --n_heads $n_heads \
                      --d_model $d_model \
                      --d_ff $(($d_model * 2))  \
                      --enc_in 1 \
                      --c_out 1 \
                      --dropout 0.2 \
                      --fc_dropout 0.2 \
                      --train_epochs 50 \
                      --patience  10 \
                      --lradj 'TST' \
                      --pct_start 0.2 \
                      --batch_size 32 \
                      --learning_rate $lr \
                      --target "load" \
                      --itr 1 \

                done
            done
        done
    done
done


#for pred_len in 192; do
#    for layers in "3" "6"; do
#        for hpo in "64 8" "128 16" "256 16"; do
#            for seq_len in 336 512; do
#                for lr in 0.001 0.0005; do