#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import jieba

from tqdm import tqdm
import numpy as np

from rouge_chinese import Rouge

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, HfArgumentParser
from transformers.generation.utils import GenerationConfig

from dataclasses import dataclass, field
from typing import Optional



@dataclass
class PathArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_path: str = field(
        default="",
        metadata={"help": "模型的路径"}
    )

    test_file: Optional[str] = field(
        default="", 
        metadata={"help": "测试集数据"}
    )
    output_dir: Optional[str] = field(
        default="./result/",
        metadata={"help": "输出文件夹"},
    )


def main():

    parser = HfArgumentParser(PathArguments)
    path_args = parser.parse_args_into_dataclasses()[0]

    rouge = Rouge()

    # 保存所有的分值
    score_dict = {
        "rouge-1-p": [],
        "rouge-1-r": [],
        "rouge-1-f": [],
        "rouge-2-p": [],
        "rouge-2-r": [],
        "rouge-2-f": [],
        "rouge-l-p": [],
        "rouge-l-r": [],
        "rouge-l-f": []
    }

    model_dir = path_args.model_path
    test_file = path_args.test_file

    out_dir = path_args.output_dir

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    check_point_name = model_dir.split('/')[-1]

    out_file = os.path.join(out_dir, check_point_name)
    out_metrics_file = os.path.join(out_dir, "metrics_" + check_point_name)

    # do predict
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_dir)

    res = []
    with open(test_file, 'r') as f:
        for line in tqdm(f.readlines()):
            data = json.loads(line.strip())
            text = data["input"]
            label = data['label']

            # 获得模型输出
            inputs = [tokenizer.bos_token_id] + tokenizer.encode(text) + [tokenizer.eos_token_id]
            input_ids = torch.LongTensor([inputs]).to('cuda:0')
            outputs = model.generate(input_ids, generation_config=model.generation_config)
            response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            data['response'] = response

            if not response:
                response = '空'
            hypothesis = list(jieba.cut(response))
            reference = list(jieba.cut(label))

            # 计算rouge分
            score = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
            score = score[0]
            for key,val in score.items():
                for sub_key, sub_val in val.items():
                    score_dict[key+'-'+sub_key].append(sub_val)

            exam = json.dumps(data, ensure_ascii=False)   
            res.append(exam)

    # 将每个指标求平均
    for key,val in score_dict.items():
        score_dict[key] = float(np.mean(val))

    # 保存指标文件
    with open(out_metrics_file, 'w') as f:
        json.dump(score_dict, f)

    # 保存预测文件
    res = '\n'.join(res)
    with open(out_file, 'w', encoding="utf-8") as f:
        f.write(res)
    


if __name__ == "__main__":
    main()