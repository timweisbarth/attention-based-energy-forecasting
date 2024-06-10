#!/bin/bash
#SBATCH --job-name="ftM_TestSet_OtherModels"
#SBATCH --gres=gpu:1
#SBATCH --partition=2080-galvani
#SBATCH --time 0-20:00:00 #
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x-%j.out  # cannot use $WORK 
#SBATCH --error=/mnt/qb/work/ludwig/lqb853/slurm_logs/%x_%j.err

# useful for debugging
scontrol show job $SLURM_JOB_ID
nvidia-smi # only if you requested any gpus

#current_folder=$(echo "${0}" | awk -F'/' '{for(i=1; i<=NF; i++) if($i ~ /^Exp/) print $i}')
current_folder="Exp4"


#Load, LSTM
for hpos in "24 512 1" "96 512 1" "192 1024 2" "336 1024 2" "720 1024 2"; do
    read pred_len d_model layers <<< $hpos
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
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers $layers \
      --d_model $d_model \
      --learning_rate 0.0001 \
      --batch_size 32 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --target "load" \
      --itr 1 \
      --train_epochs 30 \
      --patience 3 \
      --final_run_train_on_train_and_val \

done

# Load, iTransformer
for hpos in "24 256" "96 256" "192 256" "336 512" "720 1024"; do
    read pred_len d_model <<< $hpos
    srun python3 -u run.py \
      --is_training 1 \
      --des $current_folder \
      --checkpoints ./checkpoints/$current_folder \
      --root_path ./data/preproc/ \
      --data_path smard_data_DE.csv \
      --model_id 'load' \
      --model iTransformer \
      --data smard \
      --features S \
      --seq_len 336 \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers 3 \
      --d_model $d_model \
      --d_ff $d_model \
      --n_heads 8 \
      --learning_rate 0.001 \
      --batch_size 32 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --target "load" \
      --itr 1 \
      --train_epochs 30 \
      --patience 3 \
      --final_run_train_on_train_and_val \

done      




            







