
python3 -u run.py \
  --is_training 1 \
  --root_path ./data/preproc/ \
  --data_path smard_data.csv \
  --model_id 'load' \
  --model Transformer \
  --data smard \
  --features S \
  --seq_len 48 \
  --pred_len 24 \
  --d_model 12 \
  --d_ff 48 \
  --n_heads 4 \
  --target "load" \
  --enc_in 1\
  --dec_in 1 \
  --c_out 1 \
  --itr 1 \
  --train_epochs 1\