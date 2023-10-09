
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 小说， baichuan1
deepspeed  --master_port 60010 run_train_eval.py \
      --deepspeed deepspeed.json \
      --save_step=1000 \
      --train_file=/ssd/lipengwei/songyingxin/Baichuan2-Novel/dataset/novel-finetune/train.json \
      --validation_file=/ssd/lipengwei/songyingxin/Baichuan2-Novel/dataset/novel-finetune/dev.json \
      --max_length=4002 \
      --output_model_name=xiaoshuo_baichuan_1_lora_epoch_5 \
      --num_train_epochs=5 \
      --learning_rate=1e-5 \
      --per_device_train_batch_size=2 \
      --per_device_eval_batch_size=2 \
      --gradient_accumulation_steps=32 \
      --model_path=/ssd/lipengwei/songyingxin/models/baichuan-7B

# 小说，baichuan2
deepspeed  --master_port 60010 run_train_eval.py \
      --deepspeed deepspeed.json \
      --save_step=50 \
      --train_file=/ssd/lipengwei/songyingxin/Baichuan2-Novel/dataset/novel-finetune/train.json \
      --validation_file=/ssd/lipengwei/songyingxin/Baichuan2-Novel/dataset/novel-finetune/dev.json \
      --max_length=4002 \
      --output_model_name=xiaoshuo_baichuan_2_lora_pretrain \
      --num_train_epochs=1 \
      --learning_rate=1e-5 \
      --per_device_train_batch_size=2 \
      --per_device_eval_batch_size=2 \
      --gradient_accumulation_steps=32 \
      --model_path=/data/lipengwei/songyingxin/output_pretrain/checkpoint-650

# 论文 Baichuan2
deepspeed  --master_port 60010 run_train_eval.py \
      --deepspeed deepspeed.json \
      --save_step=5 \
      --train_file=/ssd/lipengwei/songyingxin/Baichuan2-Novel/dataset/pdf-dataset/train.json \
      --validation_file=/ssd/lipengwei/songyingxin/Baichuan2-Novel/dataset/pdf-dataset/dev.json \
      --max_length=1500 \
      --output_model_name=paper_baichuan2_lora \
      --num_train_epochs=2 \
      --learning_rate=1e-5 \
      --per_device_train_batch_size=4 \
      --per_device_eval_batch_size=4 \
      --gradient_accumulation_steps=16 \
      --model_path=/ssd/lipengwei/songyingxin/models/baichuan2-7b-chat

# 论文 baichuan 1
deepspeed  --master_port 60006 run_train_eval.py \
      --deepspeed deepspeed.json \
      --save_step=10 \
      --train_file=/ssd/lipengwei/songyingxin/Baichuan2-Novel/dataset/pdf-dataset/train.json \
      --validation_file=/ssd/lipengwei/songyingxin/Baichuan2-Novel/dataset/pdf-dataset/dev.json \
      --max_length=1500 \
      --output_model_name=paper_baichuan1_lora \
      --num_train_epochs=2 \
      --learning_rate=1e-5 \
      --per_device_train_batch_size=4 \
      --per_device_eval_batch_size=4 \
      --gradient_accumulation_steps=16 \
      --model_path=/ssd/lipengwei/songyingxin/models/baichuan-7B