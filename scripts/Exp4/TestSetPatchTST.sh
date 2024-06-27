#!/bin/bash
#SBATCH --job-name="ftS_patchtst_HPO_192"
#SBATCH --gres=gpu:1
#SBATCH --partition=2080-galvani
#SBATCH --time 1-23:00:00 #
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x-%j.out  # cannot use $WORK 
#SBATCH --error=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x_%j.err

# useful for debugging
scontrol show job $SLURM_JOB_ID
nvidia-smi # only if you requested any gpus

current_folder="Exp4"

# lr: larger because training took pretty long
for pred_len in 24 96 192 336 720; do
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
      --seq_len 512 \
      --pred_len $pred_len \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256  \
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
      --final_run_train_on_train_and_val \

done