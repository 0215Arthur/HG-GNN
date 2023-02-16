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
#from pandarallel import pandarallel
import pickle
from tqdm import tqdm 
import dgl

def userTopItems(dataset,K=10):
    with open(f'./dataset/{dataset}/train.pkl','rb') as f:
        session_data=pickle.load(f)
    u_dict=dict()
    item_pop=dict()
    for uid in tqdm(session_data):
        u_sess=session_data[uid]
        for sess in u_sess:
            for vid in sess:
                item_pop.setdefault(vid,0)
                item_pop[vid]+=1
    for uid in tqdm(session_data):
        u_sess=session_data[uid]
        u_dict.setdefault(uid,dict())
        for sess in u_sess:
            for vid in sess:
                u_dict[uid].setdefault(vid,0)
                u_dict[uid][vid]+=(item_pop[vid]*0.75)
    user_topK={}
        #print(user_sim_matrix)
    for user in u_dict:
        hot_items=[key for key,value in sorted(u_dict[user].items(), key=itemgetter(1), reverse=True)[0:K]]
        cold_items=[key for key,value in sorted(u_dict[user].items(), key=itemgetter(1), reverse=False)[0:K]]
        user_topK[user]=list(set(hot_items).union(set(cold_items)))
    with open(f'./dataset/{dataset}/userTopItems.pkl','wb') as f:
            pickle.dump(user_topK,f)


def itemTopUsers(dataset,K=10):
    with open(f'./dataset/{dataset}/train.pkl','rb') as f:
        session_data=pickle.load(f)
    v_dict=dict()
    user_active=dict()
    for uid in tqdm(session_data):
        u_sess=session_data[uid]
        user_active[uid]=sum([len(sess) for sess in   u_sess])

    for uid in tqdm(session_data):
        u_sess=session_data[uid]
        
        for sess in u_sess:
            for vid in sess:
                v_dict.setdefault(vid,dict())
                v_dict[vid].setdefault(uid,0)
                v_dict[vid][uid]+=(user_active[uid]*0.75)
    item_topK={}
        #print(user_sim_matrix)
    for item in v_dict:
        hot_users=[key for key,value in sorted(v_dict[item].items(), key=itemgetter(1), reverse=True)[0:K]]
        cold_users=[key for key,value in sorted(v_dict[item].items(), key=itemgetter(1), reverse=False)[0:K]]
        item_topK[item]=list(set(hot_users).union(set(cold_users)))
    with open(f'./dataset/{dataset}/itemTopUtems.pkl','wb') as f:
            pickle.dump(item_topK,f)


def userCF(dataset):
    """
    calculate user similarity
    """
    vid_user = {}
    user_sim_matrix = {}
    uid_vcount = {}
    with open(f'./dataset/{dataset}/train.pkl', 'rb') as f:
        session_data = pickle.load(f)
    for uid in tqdm(session_data):
        u_sess = session_data[uid]
        uid_vcount.setdefault(uid, set())
        for sess in u_sess:
            for vid in sess:
                if vid not in vid_user:
                    vid_user[vid] = set()
                vid_user[vid].add(uid)
                if vid not in vid_user:
                    vid_user[vid] = set()
                vid_user[vid].add(uid)
                uid_vcount[uid].add(vid)

    for vid, users in tqdm(vid_user.items()):
        for u in users:
            for v in users:
                if u == v:
                    continue
                user_sim_matrix.setdefault(u, {})
                user_sim_matrix[u].setdefault(v, 0)
                user_sim_matrix[u][v] += (1 / len(users))
    for u, related_users in user_sim_matrix.items():
        for v, count in related_users.items():
            user_sim_matrix[u][v] = count / math.sqrt(len(uid_vcount[u]) * len(uid_vcount[v]))
    user_topK = {}
    # print(user_sim_matrix)
    for user in user_sim_matrix:
        user_topK[user] = sorted(user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[0:100]
    with open(f'./dataset/{dataset}/u2u_sim.pkl', 'wb') as f:
        pickle.dump(user_topK, f)


def itemCF(dataset):
    """
    calucate item similarity 
    """
    uid_item = {}
    item_sim_matrix = {}
    vid_ucount = {}
    with open(f'./dataset/{dataset}/train.pkl', 'rb') as f:
        session_data = pickle.load(f)
    for uid in tqdm(session_data):
        u_sess = session_data[uid]
        uid_item[uid] = set()
        # uid_vcount.setdefault(uid,set())
        for sess in u_sess:
            for vid in sess:
                uid_item[uid].add(vid)
                vid_ucount.setdefault(vid, set())
                vid_ucount[vid].add(uid)

    for uid, items in tqdm(uid_item.items()):
        for v in items:
            for _v in items:
                if _v == v:
                    continue
                item_sim_matrix.setdefault(v, {})
                item_sim_matrix[v].setdefault(_v, 0)
                item_sim_matrix[v][_v] += (1 / len(items))
    for v, related_items in item_sim_matrix.items():
        for _v, count in related_items.items():
            item_sim_matrix[v][_v] = count / math.sqrt(len(vid_ucount[v]) * len(vid_ucount[_v]))
    item_topK = {}
    # print(user_sim_matrix)
    for item in item_sim_matrix:
        item_topK[item] = sorted(item_sim_matrix[item].items(), key=itemgetter(1), reverse=True)[0:200]
    with open(f'./dataset/{dataset}/i2i_sim.pkl', 'wb') as f:
        pickle.dump(item_topK, f)


def uui_graph(dataset_name, sample_size, topK, add_u = True, add_v = True):
    """
    dataset_name:
    sample_size:
    topK:
    add_u:
    add_v:
    """
    pre = []
    nxt = []
    src_v = []
    dst_u = []
    # build i2i / u2u relations
    itemCF(dataset_name)
    userCF(dataset_name)    

    with open(f'./dataset/{dataset_name}/train.pkl', 'rb') as f:
        graph = pickle.load(f)

    with open(f'./dataset/{dataset_name}/adj_{sample_size}.pkl', 'rb') as f:
        adj = pickle.load(f)
    adj_in = adj[0]
    adj_out = adj[1]
    print('adj_in:', len(adj_in))
    print('adj_out:', len(adj_out))
    ## sample graph
    for i in range(len(adj_in)):
        if i == 0:
            continue
        _pre = []
        _nxt = []
        for item in adj_in[i]:
            _pre.append(i)
            _nxt.append(item)
        pre += _pre
        nxt += _nxt
    o_pre = []
    o_nxt = []
    for i in range(len(adj_out)):
        if i == 0:
            continue
        _pre = []
        _nxt = []
        for item in adj_out[i]:
            _pre.append(i)
            _nxt.append(item)
        o_pre += _pre
        o_nxt += _nxt

    for u in tqdm(graph, desc='build the graph...', leave=False):
        u_seqs = graph[u]
        for s in u_seqs:
            pre += s[:-1]
            nxt += s[1:]
            dst_u += [u for _ in s]
            src_v += s

    with open(f'./dataset/{dataset_name}/u2u_sim.pkl', 'rb') as f:
        u2u_sim = pickle.load(f)

    with open(f'./dataset/{dataset_name}/i2i_sim.pkl','rb') as f:
        i2i_sim=pickle.load(f)

    topv_src=[]
    topv_dst=[]
    count_v=0
    for v in tqdm(i2i_sim,desc='gen_seq...',leave=False):
        tmp_src=[]
        tmp_dst=[]

        exclusion=adj_in[v]+adj_out[v]
        for (vid,value) in i2i_sim[v][:topK][:int(len(exclusion))]:
            if vid not in exclusion:
                tmp_src.append(vid)
                tmp_dst.append(v)
        topv_src+=tmp_src
        topv_dst+=tmp_dst

    u_src = []
    u_dst = []
    for u in tqdm(u2u_sim, desc='gen_seq...', leave=False):
        tmp_src = []
        tmp_dst = []
        for (uid, value) in u2u_sim[u][:topK]:
            tmp_src.append(uid)
            tmp_dst.append(u)
        u_src += tmp_src
        u_dst += tmp_dst

    count = 0
    for i in adj_in:
        count += len(i)
    print('local ajdency-in:', count / len(adj_in))
    count = 0
    for i in adj_out:
        count += len(i)
    print('local ajdency-out:', count / len(adj_out))

    item_num = max(max(pre), max(nxt)) +1
    print('addiotn item num', item_num)
    user_num = max(max(u_src), max(u_dst))
    u_src = [u + item_num for u in u_src]
    u_dst = [u + item_num for u in u_dst]
    dst_u = [u + item_num for u in dst_u]

    G = dgl.graph((pre, nxt))
    G = dgl.add_edges(G, nxt, pre)
    G = dgl.add_edges(G, dst_u, src_v)
    G = dgl.add_edges(G, src_v, dst_u)

    if add_u:
        G = dgl.add_edges(G, u_src, u_dst)
        G = dgl.add_edges(G, u_dst, u_src)
    
    if add_v:
        G = dgl.add_edges(G, topv_src, topv_dst)
        G = dgl.add_edges(G, topv_dst, topv_src)


    G=dgl.add_self_loop(G)

    return G,item_num


def sample_relations(dataset_name, num, sample_size=20):
    """

    """
    adj1 = [dict() for _ in range(num)]
    adj2 = [dict() for _ in range(num)]
    adj_in = [[] for _ in range(num)]
    adj_out = [[] for _ in range(num)]
    relation_out = []
    relation_in = []

    with open(f'./dataset/{dataset_name}/train.pkl', 'rb') as f:
        graph = pickle.load(f)

    for u in tqdm(graph, desc='build the graph...', leave=False):
        u_seqs = graph[u]
        for s in u_seqs:
            for i in range(len(s) - 1):
                relation_out.append([s[i], s[i + 1]])
                relation_in.append([s[i + 1], s[i]])
  
    for tup in relation_out:
        if tup[1] in adj1[tup[0]].keys():
            adj1[tup[0]][tup[1]] += 1
        else:
            adj1[tup[0]][tup[1]] = 1
    for tup in relation_in:
        if tup[1] in adj2[tup[0]].keys():
            adj2[tup[0]][tup[1]] += 1
        else:
            adj2[tup[0]][tup[1]] = 1

    weight = [[] for _ in range(num)]

    for t in range(1, num):
        x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
        adj_out[t] = [v[0] for v in x]

    for t in range(1, num):
        x = [v for v in sorted(adj2[t].items(), reverse=True, key=lambda x: x[1])]
        adj_in[t] = [v[0] for v in x]

    # edge sampling 
    for i in range(1, num):
        adj_in[i] = adj_in[i][:sample_size]
    for i in range(1, num):
        adj_out[i] = adj_out[i][:sample_size]

    with open(f'./dataset/{dataset_name}/adj_{sample_size}.pkl', 'wb') as f:
        pickle.dump([adj_in, adj_out], f)

