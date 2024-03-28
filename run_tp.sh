#weights link https://huggingface.co/huggyllama/llama-7b
weight_path=/dataset/mega/llama-7b


num_gpus=8

epoch=3
# wa for train samples counting, 
#real_gbs=per_device_train_batch_size,
#counting_gbs=per_device_train_batch_size*n_gpus
wa_epoch=$(expr $num_gpus \* $epoch)
echo "wa_epoch is: $wa_epoch"

deepspeed --num_gpus $num_gpus  \
    --master_port 52335  train.py  \
    --model_name_or_path  $weight_path \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir out \
    --num_train_epochs $wa_epoch \
    --gradient_checkpointing false \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy no \
    --save_strategy steps  \
    --save_steps 2000 \
    --save_total_limit 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 0 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed "./configs/ds_config_tp.json"
