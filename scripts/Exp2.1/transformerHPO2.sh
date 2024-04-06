#!/bin/bash
#SBATCH --job-name="ftS_transformerHP02_2_1"
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

for pred_len in 96 336
do
    for hpos in "2 1 " "3 2" "4 3"
    do
        for d_model in 32 64 128 256 512 #Watch out for n_heads!
        do
            read e_layers d_layers <<< $hpos
            srun python3 -u run.py \
              --is_training 1 \
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
              --factor 3 \
              --enc_in 1 \
              --dec_in 1 \
              --c_out 1 \
              --target "load" \
              --itr 1 \

        done
    done
done