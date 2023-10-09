#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import jieba

from tqdm import tqdm
import numpy as np

from rouge_chinese import Rouge

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.generation.utils import GenerationConfig
from peft import PeftModel

def main():

    rouge = Rouge()
    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": []
    }
    # model_dir = "/ssd/lipengwei/songyingxin/models/baichuan2-7b-chat"
    model_dir = '/ssd/lipengwei/songyingxin/models/baichuan-7B'
    peft_model_dir = '/data/lipengwei/songyingxin/output_finetune/xiaoshuo_baichuan_1_lora_epoch_5'

    test_file = "/ssd/lipengwei/songyingxin/Baichuan2-Novel/dataset/novel-finetune/dev.json"

    out_dir = './result/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    check_point_name = peft_model_dir.split('/')[-1]
    out_file = out_dir + check_point_name
    out_metrics_file = out_dir + "metrics_" + check_point_name

    # do predict
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_dir)

    model_lora = PeftModel.from_pretrained(
        model,
        peft_model_dir
    ).to('cuda:0')

    res = []
    with open(test_file, 'r') as f:
        for line in tqdm(f.readlines()[:200]):
            data = json.loads(line.strip())
            text = data["input"]
            label = data['label']

            # 获得模型输出
            inputs = [tokenizer.bos_token_id] + tokenizer.encode(text) + [tokenizer.eos_token_id]
            input_ids = torch.LongTensor([inputs]).to('cuda:0')
            inputs = {
                'input_ids': input_ids
            }
            outputs = model_lora.generate(**inputs, generation_config=model.generation_config)
            response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            data['response'] = response

            if not response:
                response = '空'
            hypothesis = list(jieba.cut(response))
            reference = list(jieba.cut(label))

            score = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
            score = score[0]
            for key,val in score.items():
                score_dict[key].append(round(val["f"] * 100, 4))

            exam = json.dumps(data, ensure_ascii=False)   
            res.append(exam)

    for key,val in score_dict.items():
        score_dict[key] = float(np.mean(val))

    print(score_dict)
    with open(out_metrics_file, 'w') as f:
        json.dump(score_dict, f)

    res = '\n'.join(res)

    with open(out_file, 'w', encoding="utf-8") as f:
        f.write(res)

if __name__ == "__main__":
    main()