
echo "${0##*x/}"
echo "${0}" | awk -F'/' '{for(i=1; i<=NF; i++) if($i ~ /^Exp/) print $i}'

current_folder=$(echo "${0}" | awk -F'/' '{for(i=1; i<=NF; i++) if($i ~ /^Exp/) print $i}')
echo "hello/$current_folder"

export CUDA_VISIBLE_DEVICES=0


python3 -u run.py \
  --is_training 1 \
  --des Test/$current_folder \
  --checkpoints ./checkpoints/Test/$current_folder \
  --root_path ./data/preproc/ \
  --data_path smard_data_DE.csv \
  --model_id 'load' \
  --model Transformer \
  --data smard \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 48 \
  --e_layers 3 \
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

done


