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
        input_text = self.examples['input'][index]
        label_text = self.examples['label'][index]

        source_ids = self.tokenizer.encode(text=input_text)
        target_ids = self.tokenizer.encode(text=label_text)

        input_ids = [self.tokenizer.bos_token_id] + source_ids
        label_ids = [self.tokenizer.bos_token_id] + [self.ignore_index] * len(source_ids)

        input_ids = input_ids + [self.tokenizer.eos_token_id] + target_ids + [self.tokenizer.eos_token_id]
        label_ids = label_ids + [self.tokenizer.eos_token_id] + target_ids + [self.tokenizer.eos_token_id]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            label_ids = label_ids[:self.max_length]

        # padding 处理
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
            
        padding_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
        label_ids = label_ids + [self.ignore_index] * padding_len

        input_ids = torch.LongTensor(input_ids)
        label_ids = torch.LongTensor(label_ids)

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