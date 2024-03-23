

for pred_len in 50
do
    python3 -u run.py \
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
      --itr 1 \
      --n_heads 2 \
      --d_model 48 \
      --d_ff 48 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --train_epochs 2 \
      --patience 10 \
      --lradj 'TST' \
      --pct_start 0.2 \
      --batch_size 32 \
      --learning_rate 0.0001 \

done