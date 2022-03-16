

"""
following the torch.Dataset, we prepare the standard dataset input for Models

"""
import os
import pickle
import dgl
import json
from multiprocessing import Pool
from multiprocessing import cpu_count 
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


from utils import log_exec_time


def gen_seq(data_list):
    out_seqs=[]
    label=[]
    uid=[]
    for u in tqdm(data_list,desc='gen_seq...',leave=False):
        u_seqs=data_list[u]
        for seq in u_seqs:
            for i in range(1,len(seq)):
                uid.append(int(u))
                out_seqs.append(seq[:-i])
                label.append([seq[-i]])
    # for s in tqdm(data_list,desc='gen_seq...',leave=False):
    #     full_seq=s['1']
    #     for i in range(1,len(full_seq)):
    #         out_seqs.append(full_seq[:-i])
    #         label.append([full_seq[-i]])
    # def len_argsort(seq):
    #     return sorted(range(len(seq)), key=lambda x: len(seq[x]),reverse=True)
    # sorted_idx=len_argsort(out_seqs)
    # final_seqs=[]
    # for i in sorted_idx:
    #     final_seqs.append([uid[i],out_seqs[i],label[i]])
    return (uid,out_seqs,label)


def common_seq(data_list):
    out_seqs=[]
    label=[]
    uid=[]
    for u in tqdm(data_list,desc='gen_seq...',leave=False):
        u_seqs=data_list[u]
        for seq in u_seqs:      
            for i in range(1,len(seq)):
                uid.append(int(u))
                out_seqs.append(seq[:-i])
                label.append([seq[-i]])
    
    final_seqs=[]
    for i in range(len(uid)):
        final_seqs.append([uid[i],out_seqs[i],label[i]])
    return final_seqs


def load_data(dataset,data_path):
    
    if not os.path.exists(os.path.join(data_path,dataset)+'/train_seq.pkl'): 
            # create the tmp filepath to save data
        print('try to build ',os.path.join(data_path,dataset)+'/train_seq.pkl')
        with open(os.path.join(data_path,dataset)+'/train.pkl','rb') as f:
            train_data=pickle.load(f)
        max_vid=0
        max_uid=0
        for u in train_data:
            if u>max_uid:
                max_uid=u
            for sess in train_data[u]:
                if max_vid<max(sess):
                    max_vid=max(sess)
        

        try:
            with open(os.path.join(data_path,dataset)+'/all_test.pkl','rb') as f:
                test_data=pickle.load(f)
        except:
            with open(os.path.join(data_path,dataset)+'/test.pkl','rb') as f:
                test_data=pickle.load(f)
        train_data=common_seq(train_data)
        #train_data=common_seq(train_data)

      #  val_data=gen_seq(val_data)
        test_data=common_seq(test_data)
        

        with open(os.path.join(data_path,dataset)+'/test_seq.pkl','wb') as f:
            pickle.dump(test_data,f)

        with open(os.path.join(data_path,dataset)+'/train_seq.pkl','wb') as f:
            pickle.dump(train_data,f)
        return train_data,test_data,max_vid,max_uid

    
    with open(os.path.join(data_path,dataset)+'/train_seq.pkl','rb') as f:
        train_data=pickle.load(f)
    max_vid=0
    max_uid=0
    #print(train_data[:3])
    for data in train_data:
        if data[0]>max_uid:
            max_uid=data[0]
        if max_vid<max(data[1]):
            max_vid=max(data[1])
        if max_vid<max(data[2]):
            max_vid=max(data[2])
        

    with open(os.path.join(data_path,dataset)+'/test_seq.pkl','rb') as f:
        test_data=pickle.load(f)
    for data in test_data:
        if data[0]>max_uid:
            max_uid=data[0]
        if max_vid<max(data[1]):
            max_vid=max(data[1])
        if max_vid<max(data[2]):
            max_vid=max(data[2])

    return train_data,test_data,max_vid,max_uid

        

class SessionDataset(Dataset):
    def __init__(self, data,config,max_len=None):
        """
        args:
            config(dict):
            data_type(int): 0: train 1: val 2:test
        """
        super(SessionDataset, self).__init__()
        #self.config=config
        self.data=data
        if max_len:
            self.max_seq_len=max_len
        else:
            self.max_seq_len=config['dataset.seq_len']

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        data format:
        <[uid]> <[v1,v2,v3]> <label>
        """
     
        
        data=self.data[index]
        uid=np.array([data[0]],dtype=np.int)
        browsed_ids=np.zeros((self.max_seq_len),dtype=np.int)
        #reverse_ids=np.zeros((self.max_seq_len),dtype=np.int)
        seq_len=len(data[1][-self.max_seq_len:])
        mask=np.array([1 for i in range(seq_len)]+[ 0 for i in range(self.max_seq_len-seq_len)],dtype=np.int)
        pos_idx=np.array([seq_len-i-1 for i in range(seq_len)]+[ 0 for i in range(self.max_seq_len-seq_len)],dtype=np.int)
        browsed_ids[:seq_len]=np.array(data[1][-self.max_seq_len:])
        #reverse_ids[:seq_len]=np.array(list(reversed(data[1][-self.max_seq_len:])))
        seq_len=np.array(seq_len,dtype=np.int)
   
        label=np.array(data[2],dtype=np.int)

        return uid,browsed_ids,mask,seq_len,label,pos_idx
         


class SessionGraphDataset(Dataset):
    def __init__(self, data,config,max_len=None):
        """
        args:
            config(dict):
            data_type(int): 0: train 1: val 2:test
        """
        super(SessionGraphDataset, self).__init__()
        #self.config=config
        self.data=data
        if max_len:
            self.max_seq_len=max_len
        else:
            self.max_seq_len=config['dataset.seq_len']

    
    def __len__(self):
        return len(self.data)

  
    def __getitem__(self, index):
        """
        data format:
        <[uid]> <[v1,v2,v3]> <label>
        """
     
        
        data=self.data[index]
        uid=np.array([data[0]],dtype=np.int)
        # global_ids=np.zeros((self.max_seq_len),dtype=np.int)
        # local_ids=np.zeros((self.max_seq_len),dtype=np.int)
        
        u_input=data[1][-self.max_seq_len:]
        ## reverse browsed ids 
        u_input=list(reversed(u_input))
        

        max_n_node = self.max_seq_len
        u_input=u_input+(self.max_seq_len - len(u_input)) * [0]
        global_ids = np.array(u_input)
        node = np.unique(u_input)
        local_nodes = node.tolist() + (max_n_node - len(node)) * [0]
        adj = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]
            adj[u][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            if u == v or adj[u][v] == 4:
                continue
            adj[v][v] = 1
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3
        
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]

        seq_len=len(data[1][-self.max_seq_len:])
        mask=np.array([1 for i in range(seq_len)]+[ 0 for i in range(self.max_seq_len-seq_len)],dtype=np.int)

        global_ids = np.array(u_input)
        local_ids = alias_inputs

        seq_len=np.array(seq_len,dtype=np.int)
   
        label=np.array(data[2],dtype=np.int)


        return uid,\
               global_ids,\
               mask,\
               seq_len,\
               label,\
               np.array(local_nodes,dtype=np.int),\
               np.array(local_ids,dtype=np.int),\
               adj#,g