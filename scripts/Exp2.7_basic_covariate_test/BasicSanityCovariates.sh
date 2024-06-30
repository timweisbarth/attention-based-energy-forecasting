current_folder="Exp2.7.1"

#for model in "Informer" "Autoformer" "LSTM" "PatchTST" "Transformer" "TSMixer" "DLinear" "iTransformer"; do
#for model in "iTransformer"; do
#for model in "Informer" "Autoformer" "LSTM" "TSMixer" "DLinear" "iTransformer"; do
for model in "Autoformer"; do
  python3 -u run.py \
    --is_training 1 \
    --des $current_folder \
    --checkpoints ./checkpoints/$current_folder \
    --root_path ./data/preproc/ \
    --data_path covariates.csv \
    --model_id 'load' \
    --model $model \
    --data smard \
    --features MS \
    --seq_len 24 \
    --label_len 24 \
    --pred_len 24 \
    --e_layers 3 \
    --d_layers 3 \
    --d_model 64 \
    --d_ff  64 \
    --n_heads 4 \
    --learning_rate 0.0005 \
    --batch_size 32 \
    --factor 3 \
    --enc_in 3 \
    --dec_in 3 \
    --c_out 1 \
    --target "load" \
    --itr 1 \
    --train_epochs 5 \
    --patience 3 \
    --dropout 0.0 \
  
done