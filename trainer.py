from utils.hparams import HParam
from utils.generate import generateText, is_word 
from utils.train import train
from utils import tokenization_bert_word_level as tokenization_bert
from dataset.dataloader import createDataloader

import os
import torch
import argparse
import numpy as np
import random


import transformers
from transformers import GPT2LMHeadModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',default='config/default.yaml', type=str, required=False, help='设置配置文件')
    parser.add_argument('-m','--model',required=True, help='model name')

    args = parser.parse_args()
    hp = HParam(args.config)
    with open(args.config, 'r') as f:
    #存储超参数为string
        hp_str = ''.join(f.readlines())

    work_path = os.path.join(hp.outputs.output_dir, args.model)
    if not os.path.exists(work_path):
        os.mkdir(work_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = tokenization_bert.BertTokenizer(vocab_file = hp.tokenizer.tokenizer_path)
    #use brand new mode 
    model_config = transformers.GPT2Config.from_json_file(hp.model.model_config)
    print('config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx

    #model = transformers.GPT2LMHeadModel(config=model_config)
    model = GPT2LMHeadModel.from_pretrained(hp.model.model_path)
    model.to(device)
    model.train()

    n_ctx= model.config.n_ctx

    train_set= createDataloader(hp, tokenizer, n_ctx)

    if not os.path.exists(hp.outputs.save_samples_path):
        os.makedirs(hp.outputs.save_samples_path)

    
    model_save_path = os.path.join(hp.outputs.output_dir, args.model, 'model_saved')

    train(hp = hp,
            train_set=train_set,
            model=model,
            model_save_path=model_save_path,
            )
    
    print('train finished')

if __name__=="__main__":
    main()

    
