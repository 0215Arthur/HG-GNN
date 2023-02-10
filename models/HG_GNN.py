import dgl
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import torchsnooper
import pickle
import dgl.function as FN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HG_GNN(nn.Module):
    # @torchsnooper.snoop()
    def __init__(self, G, config,item_num, max_seq_len=10, max_sess = 10):
        super().__init__()
        self.G = G.to(device)
        self.max_sess = max_sess
        #print(config['hidden_size'], type(config['hidden_size']))
        self.hidden_size = config['hidden_size']
        self.em_size = config['embed_size']
        self.pos_embedding = nn.Embedding(200, self.em_size)
        #self.u2e = nn.Embedding(G.number_of_nodes('user'), self.em_size).to(device)
        self.v2e = nn.Embedding(G.number_of_nodes(), self.em_size).to(device)

        self.conv1 = dglnn.SAGEConv(self.em_size, self.em_size, 'mean')
        # self.conv2 = dglnn.SAGEConv(self.em_size, self.em_size, 'mean')

        dropout = config["dropout"]
        self.emb_dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(self.em_size, self.hidden_size, 1)
        self.max_seq_len = max_seq_len
        self.W = nn.Linear(self.em_size, self.em_size)

        # node embedding
        self.linear_one = nn.Linear(self.em_size, self.em_size, bias=True)
        self.linear_two = nn.Linear(self.em_size, self.em_size, bias=True)
        self.linear_three = nn.Linear(self.em_size, 1, bias=False)

        # gru embedding
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)

        self.ct_dropout = nn.Dropout(dropout)
        
        self.user_transform = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(self.em_size, self.em_size, bias=True)
            # nn.BatchNorm1d(predict_em_size, momentum=0.5),
        )

        self.gru_transform = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(self.hidden_size * 2, self.em_size, bias=True)
            # nn.BatchNorm1d(predict_em_size, momentum=0.5),
        )

        self.sigmoid_concat = nn.Sequential(
            nn.Linear(self.em_size * 2, 1, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm1d(predict_em_size, momentum=0.5),
        )

        self.w_1 = nn.Parameter(torch.Tensor(2 * self.em_size, self.em_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.em_size, 1))
        self.glu1 = nn.Linear(self.em_size, self.em_size)
        self.glu2 = nn.Linear(self.em_size, self.em_size, bias=False)

        self.w_3 = nn.Parameter(torch.Tensor(self.em_size, self.em_size))
        self.w_4 = nn.Parameter(torch.Tensor(self.em_size, 1))
        self.glu3 = nn.Linear(self.em_size, self.em_size)
        self.glu4 = nn.Linear(self.em_size, self.em_size, bias=False)

        self.reset_parameters()

        self.item_num = item_num


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.em_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def compute_hidden_vector(self, hidden, mask, pos_idx):
        """
        hidden:
        mask:
        pos_idx:
        """
        mask = mask.float().unsqueeze(-1)
        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding(pos_idx)
        tmp = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = tmp.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        return select, tmp


    def sess_user_vector(self, user_vec, note_embeds,mask):
        """
        user_vec:
        note_embeds:
        mask: 
        """
        mask = mask.float().unsqueeze(-1)           #[Bs, L, 1]
        hs = user_vec.repeat(1, mask.shape[1], 1)   #[Bs, L, em_size]
        nh = torch.matmul(note_embeds, self.w_3)    #[Bs, L, em_size]
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu3(nh) + self.glu4(hs))
        beta = torch.matmul(nh, self.w_4)
        beta = beta * mask
        select = torch.sum(beta * note_embeds, 1)  #[Bs, em_size]

        return select


    def forward(self, user, seq, mask, seq_len, pos_idx):
        """
        seq(bs*L)
        seq: bs*L
        his_ids: bs * M
        mask:
        seq_len(bs)
        """
        user = user + self.item_num

        # HG-GNN

        h1 = self.conv1(self.G,
                        self.emb_dropout(self.v2e(torch.arange(0, self.G.number_of_nodes()).long().to(device))))

        h1 =F.relu(h1)

        bs = seq.size()[0]

        L = seq.size()[1]

        node_list = seq

        item_embeds = ( h1[node_list] + self.v2e(node_list)) / 2

        user_embeds = ( h1[user] + self.v2e(user)) / 2

        node_embeds = item_embeds.view((bs, L, -1))
        
        # lengths = seq_len.to(torch.device('cpu'))

        # a_lengths, idx = lengths.sort(0, descending=True)

        # _, un_idx = torch.sort(idx.to(torch.device('cuda')), dim=0)

        # seq = seq[idx]

        # embs = self.emb_dropout(self.v2e(seq))

        # embs = pack_padded_sequence(embs, a_lengths, batch_first=True)  #

        # gru_out, hidden = self.gru(embs)

        # gru_out, lengths = pad_packed_sequence(gru_out, batch_first=True, total_length=self.max_seq_len)

        # gru_out = torch.index_select(gru_out, 0, un_idx)

        # user_embeds = self.user_transform(user_embeds.squeeze())

        # gru_ht = hidden[-1]

        # gru_ht = torch.index_select(gru_ht, 0, un_idx)

        # c_global = gru_ht
        # q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())  ##len*bs*hidden_size
        # q2 = self.a_2(gru_ht)  
        
        # q2_expand = q2.unsqueeze(1).expand_as(q1)
        # q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand
        # # print(q2_expand.size(),q2_masked.size(),q1.size())

        # alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(mask.size())
        # c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        # c_t = torch.cat([c_local, c_global], 1)
        # gru_ht = self.gru_transform(c_t)

        seq_embeds = user_embeds.squeeze(1)  # [bs, em_size]

        sess_vec, avg_sess = self.compute_hidden_vector(node_embeds, mask, pos_idx)  # [bs, em_size]

        sess_user = self.sess_user_vector(user_embeds, node_embeds, mask)  # [bs, em_size]

        alpha = self.sigmoid_concat(torch.cat([sess_vec, sess_user], 1))  #[bs, 1]

        seq_embeds +=  (alpha * sess_vec + (1 - alpha) * sess_user)  
 
        item_embs = self.v2e.weight[1:]  

        scores = torch.matmul(seq_embeds, item_embs.permute(1, 0))      #[bs, item_num]

        return scores

     