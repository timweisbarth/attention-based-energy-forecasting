export CUDA_VISIBLE_DEVICES=0


python3 -u run.py \
  --is_training 1 \
  --root_path ./data/preproc/ \
  --data_path smard_data.csv \
  --model_id 'load' \
  --model DLinear \
  --data smard \
  --features S \
  --seq_len 96 \
  --pred_len 48 \
  --enc_in 1 \
  --target "load" \
  --itr 1 \
  --train_epochs 1\