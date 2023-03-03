from utils.adabound import AdaBound


import torch.optim as optim
from tqdm import tqdm
import torch
import os
def train(hp, train_set, model, model_save_path,device ): 

    if hp.train.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(),
                             lr=hp.train.adabound.initial,
                             final_lr=hp.train.adabound.final)
    elif hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=hp.train.adam)
    elif hp.train.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),lr = hp.train.sgd) 
    else:
        raise Exception("%s optimizer not supported" % hp.train.optimizer)

    for c_e in range(hp.train.epoch):
        print('epoch: {}'.format(c_e+1))
        loss_sum = 0
        with tqdm(desc = 'Epoch %d - train'%(c_e+1), unit = 'it', total = len(train_set)) as pbar:
            for seq_start, labels in train_set:
                seq_start = seq_start.to(device)
                labels = labels.to(device)

                #print(seq_start.shape,labels.shape)
                seq_predict = model.forward(input_ids = seq_start, labels = labels)
                #loss func
                loss, pred = seq_predict[:2]

                optimizer.zero_grad()

                loss_sum += loss
                # loss backward
                loss.backward() 
                optimizer.step()

                # 解决梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.train.max_grad_norm)

                pbar.set_postfix(loss = loss)

                pbar.update()

        print('training loss {}'.format(loss_sum))
        print('saving model for epoch {}'.format(c_e + 1))
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(os.path.join(model_save_path, 'model_epoch{}'.format(c_e + 1)))
        # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
        # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
        print('epoch {} finished'.format(c_e + 1))

        # optimizer step