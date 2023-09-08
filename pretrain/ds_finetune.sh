
export CUDA_VISIBLE_DEVICES=0

deepspeed  --master_port 60005 run_train_eval.py \
      --deepspeed deepspeed.json \
      --save_step=100 \
      --train_file=./data_examples/train.json \
      --validation_file=./data_examples/dev.json \
      --max_length=4002 \
      --output_model_name=xiaoshuo_pretrain \
      --num_train_epochs=2 \
      --learning_rate=1e-5 \
      --per_device_train_batch_size=2 \
      --per_device_eval_batch_size=2 \
      --gradient_accumulation_steps=32 \
      --model_path=百川模型地址
