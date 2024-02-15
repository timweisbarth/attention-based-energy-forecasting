export CUDA_VISIBLE_DEVICES=0


python3 -u run.py \
  --is_training 1 \
  --root_path ./data/preproc/ \
  --data_path smard_data.csv \
  --model_id 'load' \
  --model Transformer \
  --data smard \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 24 \
  --d_ff 48 \
  --target "load" \
  --itr 1 \
  --train_epochs 1\