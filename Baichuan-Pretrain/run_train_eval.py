#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import (
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq,
    set_seed,
    HfArgumentParser
)

from arguments import PathArguments, TrainerArguments
from data_utils import GPTDataset, sequential_dataloader

from transformers import Seq2SeqTrainer

def main():

    # 参数设置
    parser = HfArgumentParser((PathArguments, TrainerArguments))
    path_args, train_args = parser.parse_args_into_dataclasses()
    path_args.output_dir = os.path.join(path_args.output_dir, path_args.output_model_name)
    path_args.logging_dir = os.path.join(path_args.logging_dir, path_args.output_model_name)

    training_args = Seq2SeqTrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir=path_args.output_dir,
        logging_dir=path_args.logging_dir,
        evaluation_strategy="steps",
        learning_rate=train_args.learning_rate,
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        per_device_eval_batch_size=train_args.per_device_eval_batch_size,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        num_train_epochs=train_args.num_train_epochs,
        eval_steps=train_args.save_step,
        save_steps=train_args.save_step,
        logging_steps=train_args.save_step,
        seed=train_args.seed,
        save_total_limit=5,
        overwrite_output_dir=True,
        deepspeed=train_args.deepspeed,
        local_rank=train_args.local_rank,
        fp16=True
    )

    print("Process rank: {}; device: {}; n_gpu: {}".format(training_args.local_rank, training_args.device, training_args.n_gpu))

    # 设置随机种子
    set_seed(training_args.seed)

    # 加载数据
    data_files = {}
    if path_args.train_file is not None:
        data_files["train"] = path_args.train_file
    if path_args.validation_file is not None:
        data_files["validation"] = path_args.validation_file

    raw_datasets = load_dataset(
        "json",
        data_files=data_files
    )

    train_data = raw_datasets['train']
    dev_data = raw_datasets['validation']

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        path_args.model_path,
        trust_remote_code=True
    )

    # 生成 Dataset
    train_dataset = GPTDataset(train_args, train_data, tokenizer)
    dev_dataset = GPTDataset(train_args, dev_data, tokenizer)

    # 加载模型配置
    config = AutoConfig.from_pretrained(path_args.model_path, trust_remote_code=True)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(path_args.model_path, trust_remote_code=True, config=config)

    # 半精度训练
    model = model.half()

    # Data Collator
    label_pad_token_id = tokenizer.unk_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        padding=False
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = train_args.max_length

    # 初始化 Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # 开始训练    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    train_result = trainer.train()

    # 保存模型
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()