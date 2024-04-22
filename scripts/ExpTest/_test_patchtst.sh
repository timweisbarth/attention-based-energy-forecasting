

current_folder=$(echo "${0}" | awk -F'/' '{for(i=1; i<=NF; i++) if($i ~ /^Exp/) print $i}')


python3 -u run.py \
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
  --label_len 24 \
  --pred_len 24 \
  --e_layers 3 \
  --d_layers 3 \
  --d_model 64 \
  --d_ff 128 \
  --n_heads 4 \
  --learning_rate 0.001 \
  --batch_size 32 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --target "load" \
  --itr 1 \
  --train_epochs 5 \