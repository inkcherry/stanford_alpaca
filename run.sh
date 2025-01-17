#weights link https://huggingface.co/huggyllama/llama-7b
weight_path=/host/ssd/hf_models/llama-2-7b-hf
export WANDB_MODE=disabled
num_gpus=8
epoch=3
mbs=2
MODE=${1:-zero} 
if [ "$MODE" == "zero1tp" ]; then
  ZERO_STAGE=1
  AUTOTP_SIZE=4
  per_device_train_batch_size=$((mbs * AUTOTP_SIZE))
elif [ "$MODE" == "zero1" ]; then
  ZERO_STAGE=1
  AUTOTP_SIZE=0
  per_device_train_batch_size=$mbs
elif [ "$MODE" == "zero3" ]; then
  ZERO_STAGE=3
  AUTOTP_SIZE=0
  per_device_train_batch_size=$mbs
elif [ "$MODE" == "tp" ]; then
  ZERO_STAGE=0
  AUTOTP_SIZE=8
  per_device_train_batch_size=$((mbs * AUTOTP_SIZE))
else
  echo "error '$MODE',please use 'zero' or 'tp'。"
  exit 1
fi
TEMPLATE_FILE="configs/ds_config_temp.json"
OUTPUT_FILE="configs/ds_config.json"
sed -e "s/\${zero_stage}/${ZERO_STAGE}/g" \
    -e "s/\${autotp_size}/${AUTOTP_SIZE}/g" \
    $TEMPLATE_FILE > $OUTPUT_FILE
deepspeed --num_gpus $num_gpus  \
    --master_port 51335  train.py  \
    --model_name_or_path  $weight_path \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir out \
    --num_train_epochs $epoch \
    --gradient_checkpointing false \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy no \
    --save_strategy steps  \
    --save_steps 5000 \
    --gradient_accumulation_steps 4 \
    --learning_rate 0 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed "./configs/ds_config.json"