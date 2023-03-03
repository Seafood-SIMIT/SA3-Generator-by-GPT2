from utils.hparams import HParam
from utils.generate import generateText, is_word 
from utils import tokenization_bert_word_level as tokenization_bert

import os
import torch
import argparse

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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = hp.train.batch_size

    tokenizer = tokenization_bert.BertTokenizer(vocab_file = hp.tokenizer.tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(hp.test.model_path)
    model.to(device)
    model.eval()

    n_ctx= model.config.n_ctx

    length = hp.out.length
    if not os.path.exists(hp.outputs.save_samples_path):
        os.makedirs(hp.outputs.save_samples_path)

    samples_file = open(hp.outputs.save_samples_path + '/samples.txt','w', encoding='utf8')

    cnt_generate = 0
    while True:
        raw_text = hp.data.raw_data

        context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
        
        out = generateText(
                n_ctx = n_ctx,
                model = model,
                context=context_tokens,
                length = length,
                is_fast_pattern=hp.out.fast_pattern,
                tokenizer = tokenizer,
                temperature = hp.out.temperature,top_k=hp.out.topk, top_p=hp.out.topp,repitition_penalty = hp.out.repetition_penalty, device = device,
                )

        for i in range(batch_size):
            cnt_generate+=1
            text = tokenizer.convert_ids_to_tokens(out)

            for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
                    if is_word(item) and is_word(text[i + 1]):
                        text[i] = item + ' '
            for i, item in enumerate(text):
                if item == '[MASK]':
                    text[i] = ''
                elif item == '[CLS]':
                    text[i] = '\n\n'
                elif item == '[SEP]':
                    text[i] = '\n'
            info = "=" * 40 + " SAMPLE " + str(cnt_generate) + " " + "=" * 40 + "\n"
            print(info)
            text = ''.join(text).replace('##', '').strip()
            print(text)
            if hp.outputs.store:
                samples_file.write(info)
                samples_file.write(text)
                samples_file.write('\n')
                samples_file.write('=' * 90)
                samples_file.write('\n' * 2)
        print("=" * 80)
        break
    samples_file.close()

if __name__=="__main__":
    main()

    
