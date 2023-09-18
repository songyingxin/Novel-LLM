
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 论文 baichuan 1
deepspeed  --master_port 60004 run_train_eval.py \
      --deepspeed deepspeed.json \
      --save_step=10 \
      --train_file=pdf-dataset/train.json \
      --validation_file=pdf-dataset/dev.json \
      --max_length=1500 \
      --output_model_name=paper_baichuan1 \
      --num_train_epochs=2 \
      --per_device_train_batch_size=4 \
      --per_device_eval_batch_size=4 \
      --gradient_accumulation_steps=16 \
      --model_path=baichuan-7B
