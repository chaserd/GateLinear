
model_name=GateLinear


root_path_name=./dataset/
data_path_name=weather/weather.csv
model_id_name=weather
data_name=custom

seq_len=96
for pred_len in 96
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
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 64 --learning_rate 0.001 --random_seed $random_seed
done
done


root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

seq_len=96
for pred_len in 96
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
      --e_layers 3 \
      --d_model 512 \
      --d_ff 512 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 16 --learning_rate 0.005 --random_seed $random_seed
done
done


model_name=CrossGNN


root_path_name=./dataset/
data_path_name=weather/weather.csv
model_id_name=weather
data_name=custom

seq_len=96
for pred_len in 96 
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
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 64 --learning_rate 0.001 --random_seed $random_seed
done
done





root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

seq_len=96
for pred_len in 96
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
      --train_epochs 30 \
      --e_layers 3 \
      --patience 5 \
      --itr 1 --batch_size 16 --learning_rate 0.005 --random_seed $random_seed
done
done



model_name=DLinear


root_path_name=./dataset/
data_path_name=weather/weather.csv
model_id_name=weather
data_name=custom

seq_len=96
for pred_len in   96 
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
      --itr 1 --batch_size 64 --learning_rate 0.001 --random_seed $random_seed
done
done




root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

seq_len=96
for pred_len in   96 
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
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 32 --learning_rate 0.003 --random_seed $random_seed
done
done





model_name=iTransformer




root_path_name=./dataset/
data_path_name=weather/weather.csv
model_id_name=weather
data_name=custom

seq_len=96
for pred_len in   96 
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
      --itr 1 --batch_size 64 --learning_rate 0.001 --random_seed $random_seed
done
done










root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

seq_len=96
for pred_len in  96
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
      --itr 1 --batch_size 16 --learning_rate 0.0005 --random_seed $random_seed
done
done





model_name=Leddam


root_path_name=./dataset/
data_path_name=weather/weather.csv
model_id_name=weather
data_name=custom

seq_len=96
for pred_len in 96
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
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 64 --learning_rate 0.001 --random_seed $random_seed
done
done





root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

seq_len=96
for pred_len in 96
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
      --train_epochs 30 \
      --e_layers 3 \
      --patience 5 \
      --itr 1 --batch_size 16 --learning_rate 0.005 --random_seed $random_seed
done
done





model_name=MSGNet




root_path_name=./dataset/
data_path_name=weather/weather.csv
model_id_name=weather
data_name=custom

seq_len=96
for pred_len in 96
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
      --c_out 21 \
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 64 --learning_rate 0.001 --random_seed $random_seed
done
done





root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

seq_len=96
for pred_len in 96
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
      --c_out 321 \
      --train_epochs 30 \
      --e_layers 3 \
      --patience 5 \
      --itr 1 --batch_size 16 --learning_rate 0.005 --random_seed $random_seed
done
done





model_name=PatchTST


root_path_name=./dataset/
data_path_name=weather/weather.csv
model_id_name=weather
data_name=custom

seq_len=96
for pred_len in   96
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
for pred_len in   96
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




model_name=XLinear




root_path_name=./dataset/
data_path_name=weather/weather.csv
model_id_name=weather
data_name=custom

seq_len=96
for pred_len in 96
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
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 64 --learning_rate 0.001 --random_seed $random_seed
done
done





root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

seq_len=96
for pred_len in 96 
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
      --train_epochs 30 \
      --e_layers 3 \
      --patience 5 \
      --itr 1 --batch_size 16 --learning_rate 0.005 --random_seed $random_seed
done
done