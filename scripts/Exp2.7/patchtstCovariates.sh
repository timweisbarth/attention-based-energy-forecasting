#!/bin/bash
#SBATCH --job-name="ftS_transformerHPO_2_7"
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
current_folder="Exp2.7"

srun python3 -u run.py \
  --is_training 1 \
  --des $current_folder \
  --checkpoints ./checkpoints/$current_folder \
  --root_path ./data/preproc/ \
  --data_path covariates.csv \
  --model_id 'load' \
  --model PatchTST \
  --data smard \
  --features MS \
  --seq_len 24 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 3 \
  --d_layers 3 \
  --d_model 128 \
  --d_ff  256 \
  --n_heads 16 \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target "load" \
  --itr 1 \