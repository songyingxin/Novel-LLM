#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch

from baichuan.modeling_baichuan import BaichuanForCausalLM
from transformers import AutoTokenizer
from transformers.generation.utils import GenerationConfig

def main():

    # 模型路径
    model_dir = './baichuan'
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

    model = BaichuanForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_dir)
    for i in range(100):
        text = '根据标题生成中文学位论文摘要：股票市场预测的文本挖掘技术和系统实现'

        inputs = [tokenizer.bos_token_id] + tokenizer.encode(text) + [tokenizer.eos_token_id]
        input_ids = torch.LongTensor([inputs])
        input_ids = input_ids.to('cuda:0')
        outputs = model.generate(input_ids, generation_config=model.generation_config)
        response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        print(response)




if __name__ == "__main__":
    main()