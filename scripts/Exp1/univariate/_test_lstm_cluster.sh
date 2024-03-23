#!/bin/bash
#SBATCH --job-name="single_a100_job_for_test"
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-galvani
#SBATCH --time 0-00:30:00 # set maximum allowed runtime to 10min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/qb/work/ludwig/lqb853/%x-%j.out  # note one cannot use env variables like $WORK in #SBATCH statements

# useful for debugging
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested any gpus

ls $WORK # not necessary just here to illustrate that $WORK is avaible here

# add possibly other setup code here, e.g.
# - copy singularity images or data to local storage like /scratch_local
# - loads virtual envs, like with anaconda
# - set environments variables
# - determine commandline arguments for `srun` calls


for pred_len in 50 200
do
   srun python3 -u run.py \
     --is_training 1 \
     --root_path ./data/preproc/ \
     --data_path smard_data.csv \
     --model_id '' \
     --model LSTM \
     --data smard \
     --features M \
     --seq_len 96 \
     --label_len 48 \
     --pred_len $pred_len \
     --e_layers 2 \
     --d_layers 1 \
     --factor 3 \
     --enc_in 3 \
     --dec_in 3 \
     --c_out 3 \
     --target "load" \
     --itr 3 \
     --d_model 48 \
     --train_epochs 3 \

done

for pred_len in 50 200
do
    srun python3 -u run.py \
      --is_training 1 \
      --root_path ./data/preproc/ \
      --data_path smard_data.csv \
      --model_id '' \
      --model TSMixer \
      --data smard \
      --features M \
      --seq_len 336 \
      --pred_len $pred_len \
      --e_layers 4 \
      --enc_in 3 \
      --c_out 3 \
      --target "load" \
      --itr 3 \
      --dropout 0.5 \
      --patience 10 \
      --batch_size 32 \
      --learning_rate 0.0001 \
      --d_model 48 \
      --d_ff 48 \
      --train_epochs 3 \

done

for pred_len in 50 200
do
    srun python3 -u run.py \
      --is_training 1 \
      --root_path ./data/preproc/ \
      --data_path smard_data.csv \
      --model_id '' \
      --model PatchTST \
      --data smard \
      --features M \
      --seq_len 336 \
      --pred_len $pred_len \
      --e_layers 3 \
      --enc_in 3 \
      --c_out 3 \
      --target "load" \
      --itr 3 \
      --n_heads 2 \
      --d_model 48 \
      --d_ff 48 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --train_epochs 3 \
      --patience 10 \
      --lradj 'TST' \
      --pct_start 0.2 \
      --batch_size 32 \
      --learning_rate 0.0001 \

done

for pred_len in 50 200
do
    srun python3 -u run.py \
      --is_training 1 \
      --root_path ./data/preproc/ \
      --data_path smard_data.csv \
      --model_id 'load' \
      --model DLinear \
      --data smard \
      --features S \
      --seq_len 336 \
      --pred_len $pred_len \
      --enc_in 1 \
      --batch_size 32 \
      --learning_rate 0.005 \
      --target "load" \
      --itr 3 \
      --train_epochs 100 \
      --patience 10 \
      --d_model 48 \
      --train_epochs 3 \

done