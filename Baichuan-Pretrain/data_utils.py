#! /usr/bin/env python
# -*- coding: utf-8 -*-

import json
import torch

from torch.utils.data import Dataset, SequentialSampler, DataLoader

class GPTDataset(Dataset):
    def __init__(self, args, examples, tokenizer):

        self.tokenizer = tokenizer
        self.args = args
        self.examples = examples
        self.max_length = args.max_length
        self.ignore_index = -100

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        text = self.examples['text'][index]

        source_ids = self.tokenizer.encode(text=text)

        if len(source_ids) > self.max_length - 2:
            source_ids = source_ids[:self.max_length-2]

        input_ids = [self.tokenizer.bos_token_id] + source_ids + [self.tokenizer.eos_token_id]

        # padding 处理
        padding_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
        label_ids = input_ids[:] + [self.ignore_index] * padding_len

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids,
            "labels": label_ids,
            "attention_mask": attention_mask,
        }


def sequential_dataloader(dataset, batch_size):
    """ 顺序 Datalaoder, 数据顺序与原始数据顺序相同，eval,test 时使用
    """
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size)
    return dataloader