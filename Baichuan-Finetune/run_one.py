



#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.generation.utils import GenerationConfig

def main():

    model_dir = '/ssd/lipengwei/songyingxin/models/baichuan2-7b-chat'

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_dir)
    for i in range(100):
        messages = []
        print('输入你的文本:')
        text = '你吃饭了吗'
        print('\n\n模型的输出为:')

        inputs = [tokenizer.bos_token_id] + tokenizer.encode(text) + [tokenizer.eos_token_id]
        input_ids = torch.LongTensor([inputs])
        input_ids = input_ids.to('cuda:0')
        outputs = model.generate(input_ids, generation_config=model.generation_config)
        response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        print(response)

if __name__ == "__main__":
    main()