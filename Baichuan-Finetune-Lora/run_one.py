#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch

from peft import PeftModel

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.generation.utils import GenerationConfig

def main():

    # model_dir = '/ssd/lipengwei/songyingxin/models/baichuan2-7b-chat'
    model_dir = '/ssd/lipengwei/songyingxin/models/baichuan-7B'

    # peft_model_dir = '/data/lipengwei/songyingxin/output_finetune/xiaoshuo_baichuan_1_lora'
    peft_model_dir = '/data/lipengwei/songyingxin/output_finetune/xiaoshuo_baichuan_1_lora_epoch_5'

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    model_lora = PeftModel.from_pretrained(
        model,
        peft_model_dir
    ).to('cuda:0')

    generation_config = GenerationConfig.from_pretrained(model_dir)

    for i in range(100):
        messages = []
        print('输入你的文本:')
        # text = '根据标题生成中文学位论文摘要：股票市场预测的文本挖掘技术和系统实现'
        text = input()
        print('\n\n模型的输出为:')

        inputs = [tokenizer.bos_token_id] + tokenizer.encode(text) + [tokenizer.eos_token_id]
        input_ids = torch.LongTensor([inputs]).to('cuda:0')
        inputs = {
            'input_ids': input_ids
        }
        outputs = model.generate(**inputs, generation_config=generation_config)
        response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        print(response)




if __name__ == "__main__":
    main()