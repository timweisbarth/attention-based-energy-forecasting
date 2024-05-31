export CUDA_VISIBLE_DEVICES=0


python3 -u run.py \
  --is_training 1 \
  --root_path ./data/preproc/ \
  --data_path smard_plus_weather_without_LUandAT.csv \
  --model_id 'load' \
  --model Transformer \
  --data smard_w_weather \
  --including_weather 1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 50 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 15 \
  --dec_in 15 \
  --c_out 15 \
  --d_model 24 \
  --d_ff 48 \
  --target "load_DE" \
  --learning_rate 0.0005 \
  --itr 1 \
  --train_epochs 3\
  --optim "adamW" \
  --lradj "TST" \

#  python3 -u run.py \
#  --is_training 1 \
#  --root_path ./data/preproc/ \
#  --data_path smard_data_DE.csv \
#  --model_id 'load' \
#  --model LSTM \
#  --data smard \
#  --features M \
#  --seq_len 96 \
#  --label_len 0 \
#  --pred_len 50 \
#  --e_layers 3 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 3 \
#  --dec_in 3 \
#  --c_out 3 \
#  --d_model 24 \
#  --d_ff 48 \
#  --target "load" \
#  --learning_rate 0.0005 \
#  --itr 1 \
#  --train_epochs 3\