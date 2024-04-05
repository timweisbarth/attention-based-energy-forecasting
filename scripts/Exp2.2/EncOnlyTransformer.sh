#!/bin/bash
#SBATCH --job-name="ftS_enconlytransformer_2.2"
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-galvani
#SBATCH --time 0-04:00:00 #
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x-%j.out  # cannot use $WORK 
#SBATCH --error=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x_%j.err

# useful for debugging
scontrol show job $SLURM_JOB_ID
nvidia-smi # only if you requested any gpus

for pred_len in 24 96 336; do
  for seq_len in 96 192; do
    srun python3 -u run.py \
      --is_training 1 \
      --root_path ./data/preproc/ \
      --data_path smard_data_DE.csv \
      --model_id 'load' \
      --model EncOnlyTransformer \
      --data smard \
      --features S \
      --seq_len $seq_len \
      --label_len $(($seq_len / 2)) \
      --pred_len $pred_len \
      --e_layers 3 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --target "load" \
      --itr 1 
  done
done