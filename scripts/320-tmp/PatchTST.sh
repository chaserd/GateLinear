
model_name=PatchTST


root_path_name=./dataset/
data_path_name=weather/weather.csv
model_id_name=weather
data_name=custom

seq_len=96
for pred_len in   96 192 336 720 
do
for random_seed in 2027
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --cycle 144 \
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 16 --learning_rate 0.001 --random_seed $random_seed
done
done



root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

seq_len=96
for pred_len in   96 192 336 720 
do
for random_seed in 2027
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --cycle 168 \
      --e_layers 3 \
      --d_model 512 \
      --d_ff 512 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 16 --learning_rate 0.0003 --random_seed $random_seed
done
done


root_path_name=./dataset/PEMS/
data_path_name=PEMS03.npz
model_id_name=PEMS03
data_name=PEMS

seq_len=96
for pred_len in   96 192 336 720 
do
for random_seed in 2027
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 358 \
      --cycle 288 \
      --e_layers 3 \
      --d_model 512 \
      --d_ff 512 \
      --train_epochs 30 \
      --patience 5 \
      --use_revin 0 \
      --itr 1 --batch_size 16 --learning_rate 0.0003 --random_seed $random_seed
done
done


root_path_name=./dataset/PEMS/
data_path_name=PEMS04.npz
model_id_name=PEMS04
data_name=PEMS

seq_len=96
for pred_len in   96 192 336 720 
do
for random_seed in 2027
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 307 \
      --cycle 288 \
      --e_layers 3 \
      --d_model 512 \
      --d_ff 512 \
      --train_epochs 30 \
      --patience 5 \
      --use_revin 0 \
      --itr 1 --batch_size 16 --learning_rate 0.0003 --random_seed $random_seed
done
done




root_path_name=./dataset/ETT-small/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

model_name=PatchTST
seq_len=96
for pred_len in  96 192 336 720 
do
for random_seed in 2027
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 16 --learning_rate 0.01 --random_seed $random_seed
done
done

root_path_name=./dataset/ETT-small/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2


seq_len=96
for pred_len in   96 192 336 720 
do
for random_seed in 2027
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 16 --learning_rate 0.01 --random_seed $random_seed
done
done




root_path_name=./dataset/ETT-small/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

seq_len=96
for pred_len in   96 192 336 720 
do
for random_seed in 2027
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --des 'Exp' \
      --itr 1 --batch_size 16 --learning_rate 0.001 --random_seed $random_seed
done
done



root_path_name=./dataset/ETT-small/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2


seq_len=96
for pred_len in   96 192 336 720 
do
for random_seed in 2027
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --des 'Exp' \
      --itr 1 --batch_size 16 --learning_rate 0.001 --random_seed $random_seed
done
done






root_path_name=./dataset/PEMS/
data_path_name=PEMS07.npz
model_id_name=PEMS07
data_name=PEMS

seq_len=96
for pred_len in  96 192 336 720 
do
for random_seed in 2027
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 883 \
      --cycle 288 \
      --train_epochs 30 \
      --patience 5 \
      --des 'Exp' \
      --itr 1 --batch_size 16 --learning_rate 0.003 --random_seed $random_seed
done
done

root_path_name=./dataset/PEMS/
data_path_name=PEMS08.npz
model_id_name=PEMS08
data_name=PEMS


seq_len=96
for pred_len in   96 192 336 720 
do 
for random_seed in 2027
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 170 \
      --cycle 288 \
      --train_epochs 30 \
      --use_revin 1 \
      --patience 5 \
      --des 'Exp' \
      --itr 1 --batch_size 16 --learning_rate 0.003 --random_seed $random_seed
done
done

root_path_name=./dataset/Solar/
data_path_name=solar_AL.txt
model_id_name=Solar
data_name=Solar


seq_len=96
for pred_len in  96 192 336 720
do
for random_seed in 2027
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 137 \
      --cycle 144 \
      --train_epochs 30 \
      --patience 5 \
      --des 'Exp' \
      --itr 1 --batch_size 16 --learning_rate 0.003 --random_seed $random_seed
done
done


root_path_name=./dataset/traffic/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom


seq_len=96
for pred_len in  96 192 336 720
do
for random_seed in 2027
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --cycle 168 \
      --train_epochs 30 \
      --patience 5 \
      --des 'Exp' \
      --itr 1 --batch_size 16 --learning_rate 0.003 --random_seed $random_seed
done
done



root_path_name=./dataset/exchange_rate/
data_path_name=exchange_rate.csv
model_id_name=exchange_rate
data_name=custom


seq_len=96
for pred_len in   96 192 336 720
do
for random_seed in 2027
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 8 \
      --cycle 168 \
      --train_epochs 30 \
      --patience 5 \
      --des 'Exp' \
      --itr 1 --batch_size 8 --learning_rate 0.005 --random_seed $random_seed
done
done



# root_path_name=./dataset/
# data_path_name=national_illness.csv
# model_id_name=national_illness
# data_name=custom

# random_seed=2024
# seq_len=96
# for pred_len in 12 24 48  96 192 336 720
# do
#     python -u run.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path $data_path_name \
#       --model_id $model_id_name_$seq_len'_'$pred_len \
#       --model $model_name \
#       --data $data_name \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 7 \
#       --e_layers 3 \
#       --n_heads 4 \
#       --d_model 16 \
#       --d_ff 128 \
#       --dropout 0.3\
#       --fc_dropout 0.3\
#       --head_dropout 0\
#       --patch_len 24\
#       --cycle 168 \
#       --stride 2\
#       --des 'Exp' \
#       --train_epochs 100\
#       --patience 5 \
#       --lradj 'constant'\
#       --itr 1 --batch_size 16 --learning_rate 0.0025 
# done
