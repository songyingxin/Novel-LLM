



#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def main():

    model_dir = ""

    model_dir = ""

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(model_dir, config=config, trust_remote_code=True)
    model = model.half().cuda()
    model.eval()
    for i in range(100):    
        print("输入你想问的：")
        prompt_text = input()
        
        print("用户的输入是：{}".format(prompt_text))

        print('\n\n')
        
        print('模型的输出位：\n')
        inputs = tokenizer(prompt_text, return_tensors='pt')
        inputs = inputs.to('cuda:0')
        pred = model.generate(**inputs, max_new_tokens=4096,repetition_penalty=1.1)
        pred_text = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)

        res = prompt_text + '\n\n' + pred_text
        with open('test.txt', 'w') as f:
            f.write(res)




if __name__ == "__main__":
    main()