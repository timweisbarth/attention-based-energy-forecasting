

current_folder=$(echo "${0}" | awk -F'/' '{for(i=1; i<=NF; i++) if($i ~ /^Exp/) print $i}')


#python3 -u run.py \
#    --is_training 1 \
#    --des $current_folder \
#    --checkpoints ./checkpoints/$current_folder \
#    --root_path ./data/preproc/ \
#    --data_path covariates.csv \
#    --model_id 'load' \
#    --model TSMixer\
#    --data smard \
#    --features MS \
#    --seq_len 24 \
#    --label_len 24 \
#    --pred_len 24 \
#    --e_layers 3 \
#    --d_layers 3 \
#    --d_model 16 \
#    --d_ff  32 \
#    --n_heads 4 \
#    --learning_rate 0.0005 \
#    --batch_size 32 \
#    --factor 3 \
#    --enc_in 3 \
#    --dec_in 3\
#    --c_out 1 \
#    --target "load" \
#    --itr 1 \
#    --train_epochs 5 \
#    --patience 2 \


#current_folder="ExpTest"

# lr: larger because training took pretty long
for pred_len in 192; do
    for layers in "3" "6"; do
        for hpo in "32 4" "128 16" "256 16"; do
            for seq_len in 336 512; do
                for lr in 0.001 0.0005; do
                    read d_model n_heads <<< $hpo
                    python3 -u run.py \
                      --is_training 1 \
                      --des $current_folder \
                      --checkpoints ./checkpoints/$current_folder \
                      --root_path ./data/preproc/ \
                      --data_path smard_data_DE.csv \
                      --model_id 'load' \
                      --model PatchTST \
                      --data smard \
                      --features S \
                      --seq_len $seq_len \
                      --pred_len $pred_len \
                      --e_layers $layers \
                      --n_heads $n_heads \
                      --d_model $d_model \
                      --d_ff $(($d_model * 2))  \
                      --enc_in 1 \
                      --c_out 1 \
                      --dropout 0.2 \
                      --fc_dropout 0.2 \
                      --train_epochs 50 \
                      --patience  10 \
                      --lradj 'TST' \
                      --pct_start 0.2 \
                      --batch_size 32 \
                      --learning_rate $lr \
                      --target "load" \
                      --itr 1 \

                done
            done
        done
    done
done