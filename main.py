import pickle
import math
from operator import itemgetter
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
import pandas as pd
# from pandarallel import pandarallel
import pickle
from tqdm import tqdm
import dgl
from utils.config import Configurator

from utils.tools import get_time_dif, Logger
from data_processor.data_loader import load_data, SessionDataset
from build_graph import uui_graph, sample_relations
from data_processor.date_helper import LastFM_Process

import torch.nn.functional as F
from models import HG_GNN
import logging


config_file = 'basic.ini'  

conf = Configurator(config_file)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('loading configure from %s ...' % config_file)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("./logs/log_{}_{}_{}.txt".format(conf['recommender'], \
                                                               conf['dataset.name'], conf['comment']))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

# pre-process dataset
lp=LastFM_Process(conf)
lp._read_raw_data()
lp._split_data()


train_data, test_data, max_vid, max_uid = load_data(conf['dataset.name'], conf['dataset.path'])

print('current dataset:', conf['dataset.name'])
print('The size of train data:', len(train_data))
# print('The size of valid data',len(val_data))
print('The size of test data', len(test_data))
print('item nums:', max_vid)
# conf['dataset.n_items']=max_vid
conf.change_attr('dataset.n_items', max_vid + 1)
conf.change_attr('dataset.n_users', max_uid + 1)

print("item num {} | user num {}".format(max_vid, max_uid))


def train(config, model, device, train_iter, test_iter=None):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["lr_dc_step"], gamma=config["lr_dc"])

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0 
    dev_best_loss = float('inf')
    last_improve = 0 
    flag = False  
    AUC_best = 0
    loss_list = []
    Log = Logger(fn="./logs/log_{}_{}_{}.log".format(conf['recommender'], \
                                                     conf['dataset.name'], conf['comment']))
    best_acc = 0
    batchs = train_iter.__len__()

    for epoch in range(config['epoch']):
        epoch_t = time.time()
        print('Epoch [{}/{}]'.format(epoch + 1, config['epoch']))
        # scheduler.step() 
        loss_records = []
        L = nn.CrossEntropyLoss()
        # auc=evaluate(config,model,dev_iter,AUC_best)
        for i, (uid, browsed_ids, mask, seq_len, label, pos_idx) in enumerate(train_iter):
            model.train()
            outputs = model(uid.to(device), 
                            browsed_ids.to(device),
                            mask.to(device),
                            seq_len.to(device),
                            pos_idx.to(device)
                            )
            model.zero_grad()
            loss = L(outputs, (label - 1).to(device).squeeze())
            loss_list.append(loss.item())
            loss_records.append(loss.item())
            loss.backward()
            optimizer.step()
            STEP_SIZE = 1000
            improve = '*'

            if total_batch % STEP_SIZE == 0:
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.6},  Time: {2} {3}'
                logger.info(msg.format(total_batch, np.mean(loss_list), time_dif, improve))
                loss_list = []
            total_batch += 1
        runtime = f"\nepoch runtime : {time.time() - epoch_t:.2f}s\n"
        logger.info(runtime)

        print('preformance on test set....')
        scheduler.step()
        acc, info = evaluate_topk(config, model, test_iter)
        
        if acc > best_acc:
            best_acc = acc
            msg = f'epoch[{epoch}] test :{info}'
            Log.log(msg, red=True)
            logger.info(msg)
            last_improve = 0
            if config.save_flag:
                torch.save(model.state_dict(),
                           config["save_path"] + '/{}_epoch{}_{}.ckpt'.format(config['recommender'], \
                                                                              config['epoch'], config['comment']))
        else:
            msg = f'epoch[{epoch}] test :{info}'
            ##Log.log(msg, red=False)
            logger.info(msg)
            last_improve += 1
            if last_improve >= config['patience']:
                logger.info('Early stop: No more improvement')
                break


def metrics(res, labels):
    res = np.concatenate(res)
    acc_ar = (res == labels)  # [BS, K]
    acc = acc_ar.sum(-1)

    rank = np.argmax(acc_ar, -1) + 1
    mrr = (acc / rank).mean()
    ndcg = (acc / np.log2(rank + 1)).mean()
    return acc.mean(), mrr, ndcg


def evaluate_topk(config, model, data_iter, K=20):
    model.eval()
    hit = []
    res50 = []
    res20 = []
    res10 = []
    res5 = []
    mrr = []
    labels = []
    uids = []
    t0 = time.time()
    with torch.no_grad():
        with tqdm(total=(data_iter.__len__()), desc='Predicting', leave=False) as p:
            for i, (uid, browsed_ids, mask, seq_len, label, pos_idx) in (enumerate(data_iter)):
                # print(datas)
                outputs = model(uid.to(device), browsed_ids.to(device), mask.to(device), seq_len.to(device),
                                pos_idx.to(device)
                                # his_ids.to(device),
                                # his_mat.to(device),
                                # his_mask.to(device),
                                # his_seq_mask.to(device)
                                )
                sub_scores = outputs.topk(K)[1].cpu()
                res20.append(sub_scores)
                res10.append(outputs.topk(10)[1].cpu())
                res5.append(outputs.topk(5)[1].cpu())
                res50.append(outputs.topk(50)[1].cpu())
                labels.append(label)
                # uids.append(datas['user_id'])
                p.update(1)
    labels = np.concatenate(labels)  # .flatten()
    labels = labels - 1
    # metrics(res20,labels)
    # metrics(res10,labels)
    acc50, mrr50, ndcg50 = metrics(res50, labels)
    acc20, mrr20, ndcg20 = metrics(res20, labels)
    acc10, mrr10, ndcg10 = metrics(res10, labels)
    acc5, mrr5, ndcg5 = metrics(res5, labels)


    print("Top20 : acc {} , mrr {}, ndcg {}".format(acc20, mrr20, ndcg20))
    print("Top10 : acc {} , mrr {}, ndcg {}".format(acc10, mrr10, ndcg10))
    print("Top5 : acc {} , mrr {}, ndcg {}".format(acc5, mrr5, ndcg5))
    

    pred_time = time.time() - t0
    # acc=acc.mean()
    msg = 'Top-{} acc:{:.3f}, mrr:{:.4f}, ndcg:{:.4f}, time: {:.1f}s \n'.format(20, acc20 * 100, mrr20 * 100,
                                                                                ndcg20 * 100, pred_time)
    msg += 'Top-{} acc:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(10, acc10 * 100, mrr10 * 100, ndcg10 * 100)
    msg += 'Top-{} acc:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(5, acc5 * 100, mrr5 * 100, ndcg5 * 100)
    # msg += 'Top-{} acc:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(50, acc50 * 100, mrr50 * 100, ndcg50 * 100)

    return acc20, msg  


if __name__=="__main__":
    dgl.seed(430)
    dgl.random.seed(430)
    torch.manual_seed(430)
    torch.cuda.manual_seed_all(430)
    torch.backends.cudnn.deterministic = True
    SZ = 12
    SEQ_LEN = 10


    sample_relations(conf['dataset.name'], conf['dataset.n_items'], sample_size=SZ)

    g,item_num = uui_graph(conf['dataset.name'], sample_size=SZ, topK=20, add_u = False, add_v = False)

    print(g)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = SessionDataset(train_data, conf, max_len=SEQ_LEN)

    test_data = SessionDataset(test_data, conf, max_len=SEQ_LEN)

    train_iter = DataLoader(dataset=train_data,
                            batch_size=conf["batch_size"],
                            num_workers=4,
                            drop_last=False,
                            shuffle=True,
                            pin_memory=False)

    test_iter = DataLoader(dataset=test_data,
                        batch_size=conf["batch_size"] * 16,
                        num_workers=4,
                        drop_last=False,
                        shuffle=False,
                        pin_memory=False)

    model = HG_GNN(g, conf, item_num, SEQ_LEN).to(device)

    train(conf, model, device, train_iter, test_iter)
