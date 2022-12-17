"""
data_helper.py

    preprocess the raw datasets.(LastFM)
"""
import os
import time
import pickle
import operator
import numpy as np
import pandas as pd
from pandas import Timedelta
from tqdm import tqdm 

class BasicGraph:
    def __init__(self, min_cnt=1):
        self.min_cnt = min_cnt
        self.edge_cnt = {}
        self.adj = {}
        self._nb_edges = 0 # edges nums

    def add_edge(self, a, b):
        e = (a, b)
        self.edge_cnt.setdefault(e, 0)
        self.edge_cnt[e] += 1
        # first appear 
        if self.edge_cnt[e] == self.min_cnt:
            self.adj.setdefault(a, [])
            self.adj[a].append(b)
            self._nb_edges += 1

    def has_edge(self, a, b):
        cnt = self.edge_cnt.get((a, b), 0)
        return cnt >= self.min_cnt

    def get_edges(self):
        edges = sorted([(a, b) for (a, b), cnt in self.edge_cnt.items() if cnt >= self.min_cnt])
        return edges

    def get_adj(self, a):
        return self.adj.get(a, [])

    def nb_edges(self):
        return self._nb_edges

class Data_Process(object):
    def __init__(self,config):

        self.dataset=config['dataset.name']
        self.data_path=config['dataset.path']
        
        self.conf=config
        self.vid_map={}
        self.filter_vid={}

    def _yoochoose_item_attr(self,data_home='./raw_data/yoochoose',saved_path='./dataset'):
        category_dict={}

        df=pd.read_csv(f'{data_home}/yoochoose-clicks.dat',header=None,names=["sid", "Timestamp", "vid", "Category"],usecols=["vid", "Category"])
        new_cate=df.drop_duplicates(['vid','Category'])
        new_cate=new_cate[new_cate['Category'].isin([str(_) for _ in range(1,13)])]#['Category']#.value_counts()
        vid=new_cate['vid'].tolist()
        cate=new_cate['Category'].tolist()
        category_dict=dict(zip(vid,cate))
        with open(f'{saved_path}/yc_cate.pkl', 'wb') as f:
            pickle.dump(category_dict,f)


    def _yoochoose_(self,frac = 4,data_home='./raw_data/yoochoose',saved_path='./dataset'):
        """
        handle with the raw data format of yoochoose.
        """
        sid2vid_list = {}
        with open(f'{data_home}/yoochoose-buys.dat', 'r') as f:
            for line in tqdm(f,desc='read buy',leave=True):
                #pbar.update(1)
                #cnt += 1
                #if N > 0 and cnt > N: break
                line = line[:-1]
                sid, ts, vid, _, _ = line.split(',')
                ts = int(time.mktime(time.strptime(ts[:19], '%Y-%m-%dT%H:%M:%S')))
                sid2vid_list.setdefault(sid, [])
                sid2vid_list[sid].append([vid, 0, ts])
       
        # session_id,timestamp,item_id,category
        # 1,2014-04-07T10:51:09.277Z,214536502,0
        cnt = 0
        with open(f'{data_home}/yoochoose-clicks.dat', 'r') as f:
            for line in tqdm(f,desc='read clicks',leave=True):
                #pbar.update(1)
                cnt += 1
                #if N > 0 and cnt > N: break
                line = line[:-1]
                sid, ts, vid, cate = line.split(',')
                ts = int(time.mktime(time.strptime(ts[:19], '%Y-%m-%dT%H:%M:%S')))

                sid2vid_list.setdefault(sid, [])
                sid2vid_list[sid].append([vid, 1, ts])
        
        for sid in sid2vid_list:
            sid2vid_list[sid] = sorted(sid2vid_list[sid], key=lambda x: x[-1])

        n = len(sid2vid_list)
        # sort all sessions by the last time of the session
        yc = sorted(sid2vid_list.items(), key=lambda x: x[1][-1][-1])
        
        n_part = n // frac
        print('sid2vid len:',n)
        print('n_part:',n_part)
        yc_part = yc[-n_part:]

        out_path = f'yc_1_{frac}'
        os.mkdir(f'{saved_path}/{out_path}')
        with open(f'{saved_path}/{out_path}/data.txt', 'w') as f:
            for sid, vid_list in yc_part:
                vid_list = ','.join(map(lambda vid: ':'.join(map(str, vid)), vid_list))
                sess = ' '.join([sid, vid_list])
                f.write(sess + '\n')

    """
        def _read_diginetica(self,data_home='./raw_data/yoochoose',saved_path='./dataset'):
            diginetica raw_data:
            保留具有userid的数据
            view_df=pd.read_csv(data_home+'/train-item-views.csv',sep=';') #sessionId	userId	itemId	timeframe	eventdate
            view_df=view_df.dropna().reset_index(drop=True)
            view_df['eventdate']=view_df['eventdate'].apply(lambda x : time.mktime(time.strptime(x, '%Y-%m-%d')))

            return 
    """
    def _read_raw_data(self,dataset_name):
        """
        read raw data to select the target information.

        data format: session_id,[(vid,v_type)],session_end_time
        """
        cold_session=self.conf['dataset.filter.cold_session']
        cold_item=self.conf['dataset.filter.cold_item']

        vid_count={}
        data_list=[]

        with open(os.path.join(self.data_path,dataset_name)+'/data.txt','r') as f:
            for line in tqdm(f,desc='loading data.txt',leave=True):
                line = line[:-1]
                sid, vid_list_str = line.split()
                vid_list = []
                max_ts=0
                vid_list={}
                if len(vid_list_str.split(','))<=cold_session:
                    continue
                for vid in vid_list_str.split(','):
                    
                    vid, cls, ts = vid.split(':')
                    #cls = int(cls)  # cls: 0, 1, 2, ...
                    if cls=='1':
                        if vid not in vid_count :
                            vid_count[vid]=1
                        else:
                            vid_count[vid]+=1
                    if int(ts)>max_ts:
                        max_ts = int(ts)
                    vid_list.setdefault(cls,[])
                    vid_list[cls].append(vid)
                if len(vid_list['1'])<=cold_session:
                    continue
                data_list.append([vid,vid_list,max_ts])
                
                    #vid_list.append([vid, cls, ts]) # item_id , behavior_type , timestamp
        # sort by vid appears 
        sorted_counts = sorted(vid_count.items(), key=operator.itemgetter(1))

        filtered_data_list=[]
        for s in tqdm(data_list,desc='filterd sessions & items',leave=False):
            #print(s[1]) 0:buy 1:click
            filseq = list(filter(lambda i: vid_count[i] >= cold_item, s[1]['1']))
            if len(filseq) < 2:
                continue
            filtered_data_list.append(s)
        return filtered_data_list
        

    def _re_vindex(self,data_list,filter_flag):
        """
        reindex the items id starting from 1.

        params：
             filter_flag(bool): filter out the items not appear in the train set.

        return: only includes the vid lists

        """

        new_id=1
        new_data_list=[]
        for s in data_list:
            outseq=[]
            for vid in s[1]['1']:
                if filter_flag:
                    if vid not in self.vid_map:
                        self.filter_vid.setdefault(vid,0)
                        self.filter_vid[vid]+=1
                        continue
                                        
                if vid not in self.vid_map:
                    outseq.append(new_id)
                    self.vid_map[vid]=new_id
                    new_id+=1
                else:
                    outseq.append(self.vid_map[vid])
            new_data_list.append(outseq)
        
        #print(new_data_list[:10])
        return new_data_list
        

    def _split_data(self):
        """
        split the full dataset to train/val/test dataset.

        """
        splitter = self.conf['dataset.split']
        val_ratio = self.conf['dataset.val_ratio']

        data_list=self._read_raw_data(self.dataset)
        category_dict={}
       # print(self.data_path)
        #print(os.path.join(self.data_path,'yc_cate.pkl'))
        # with open(os.path.join(self.data_path,'yc_cate.pkl'),'rb') as f:
        #     raw_categ_dict=pickle.load(f)

        max_time=data_list[-1][-1]
        if splitter=='by_day':
            test_lens=self.conf['dataset.test_days']
            splitdate = max_time - test_lens*24*60*60
            train_sess = filter(lambda x: x[2] < splitdate, data_list)
            tes_sess = filter(lambda x: x[2] > splitdate, data_list)
            # reindex the item id
            train_sess=self._re_vindex(train_sess,filter_flag=False)
            tes_sess=self._re_vindex(tes_sess,filter_flag=True)

            val_lens= int(len(tes_sess)*val_ratio)

            val_sess=tes_sess[:val_lens]
            test_sess=tes_sess[val_lens:]
            # for key in tqdm(self.vid_map):
            #     try:
            #         category_dict[self.vid_map[key]]=raw_categ_dict[str(key)]
            #     except:
            #         print(key)
                
            
            print("session nums: train/val/test: ",len(train_sess),len(val_sess),len(tes_sess))
            
            print(train_sess[-3:],test_sess[-3:])

            print('item num:',len(self.vid_map))
        
            print('filtered item num:',len(self.filter_vid))

            train_sess={0:train_sess}
            val_sess={0:val_sess}
            test_sess={0:test_sess}
            # store the dataset with pickle
            # [vid_list_0,vid_list_1] (different behaviors,dict format)
            with open(os.path.join(self.data_path,self.dataset)+'/train.pkl','wb') as f:
                pickle.dump(train_sess,f)
            
            with open(os.path.join(self.data_path,self.dataset)+'/val.pkl','wb') as f:
                pickle.dump(val_sess,f)

            with open(os.path.join(self.data_path,self.dataset)+'/test.pkl','wb') as f:
                pickle.dump(test_sess,f)

            # with open(os.path.join(self.data_path,self.dataset)+'/cate_dict.pkl','wb') as f:
            #     pickle.dump(category_dict,f)

    def _get_sample_adj(self,G):
        random_seed=self.conf['random_seed']
        adj_size = self.conf['graph.adj_size']
        N = self.conf['dataset.n_items']
        rdm = np.random.RandomState(random_seed)
        adj=[]
        w=[]
        adj.append([0]*adj_size)
        w.append([0]*adj_size)
        for node in tqdm(range(1, N),total=N - 1, desc='building adj',leave=False):
            #pbar.update(1)
            adj_list = G.get_adj(node) # get the adjacent nodes (M nodes)
            if len(adj_list) > adj_size:
                adj_list = rdm.choice(adj_list, size=adj_size, replace=False).tolist()
            mask = [0] * (adj_size - len(adj_list))
            adj_list = adj_list[:] + mask # set the masks for padding 
            adj.append(adj_list)
            w_list = [G.edge_cnt.get((node, x), 0) for x in adj_list] # get the edge weight for each adj node of the target node. 
            w.append(w_list)
        return [adj,w]
        

    # def _build_graph(self):
    #     """"
    #     follow the specific strategy to build the graph of the sessions.

    #     return adj data.
     
    #     """
    #     with open(os.path.join(self.data_path,self.dataset)+'/train.pkl','rb') as f:
    #         train_data_list=pickle.load(f)

    #     print('build the global weighed & directed graph structure')
    #     G_in=BasicGraph()
    #     G_out=BasicGraph()
    #     for sess in tqdm(train_data_list,'build the graph',leave=False):
    #         for i,vid in enumerate(sess['1']):
    #             if i==0:
    #                 continue
    #             now_node=vid
    #             pre_node=sess['1'][i-1]
    #             if now_node!=pre_node:
    #                 G_out.add_edge(pre_node,now_node)# out degree
    #                 G_in.add_edge(now_node,pre_node) # in degree
        
    #     adj0=self._get_sample_adj(G_in)
    #     adj1=self._get_sample_adj(G_in)
    #     with open(os.path.join(self.data_path,self.dataset)+'/adj.pkl','wb') as f:
    #         pickle.dump([adj0,adj1],f)


class LastFM_Process(Data_Process):
    def __init__(self,config):
        super(LastFM_Process, self).__init__(config)
        self.interval = Timedelta(hours=8)
        self.full_df=None
    
    def _update_id(self,df, field):
        labels = pd.factorize(df[field])[0]
        kwargs = {field: labels}
        df = df.assign(**kwargs)
        return df

    def _group_sessions(self,df):
        """
        split the interation to sessions
        """
        df_prev = df.shift()
        is_new_session = (df.userId != df_prev.userId) | (
            df.timestamp - df_prev.timestamp > self.interval
        )
        session_id = is_new_session.cumsum() - 1
      
        df = df.assign(sessionId=session_id)
        return df

    def remove_immediate_repeats(self,df):
        df_prev = df.shift()
        is_not_repeat = (df.sessionId != df_prev.sessionId) | (df.itemId != df_prev.itemId)
        df_no_repeat = df[is_not_repeat]
        return df_no_repeat

    def truncate_long_sessions(self,df, max_len=20, is_sorted=False):
        if not is_sorted:
            df = df.sort_values(['sessionId', 'timestamp'])
        itemIdx = df.groupby('sessionId').cumcount()
        df_t = df[itemIdx < max_len]
        return df_t

    def keep_top_n_items(self,df, n):
        item_support = df.groupby('itemId', sort=False).size()
        top_items = item_support.nlargest(n).index
        df_top = df[df.itemId.isin(top_items)]
        return df_top
    
    def filter_short_sessions(self,df, min_len=2):
        session_len = df.groupby('sessionId', sort=False).size()
        long_sessions = session_len[session_len >= min_len].index
        df_long = df[df.sessionId.isin(long_sessions)]
        return df_long


    def filter_infreq_items(self,df, min_support=5):
        item_support = df.groupby('itemId', sort=False).size()
        freq_items = item_support[item_support >= min_support].index
        df_freq = df[df.itemId.isin(freq_items)]
        return df_freq
    
    def filter_until_all_long_and_freq(self,df, min_len=2, min_support=5):
        while True:
            df_long = self.filter_short_sessions(df, min_len)
            df_freq = self.filter_infreq_items(df_long, min_support)
            if len(df_freq) == len(df):
                break
            df = df_freq
        return df
    
    def _agg_df(self,df):
        """
        {u:[[s1],[s2],[s3],....]}
        """
        res={}
        for u,ug in tqdm(df.groupby('userId')):
            res.setdefault(u,[])
            res[u]=ug.groupby('sessionId')['itemId'].agg(list).tolist()
        return res
                 
    def _agg_all_seq(self,df):
        """
        {u:[[s1],[s2],[s3],....]}
        """
        res=[]
        for u,ug in tqdm(df.groupby('userId')):
            #res.setdefault(u,[])
            res+=ug.groupby('sessionId')['itemId'].agg(list).tolist()
        with open(os.path.join(self.data_path,self.dataset)+'/all_train_seq.txt','wb') as f:
            pickle.dump(res,f)
        return res

    def _split_data(self):
        print('split data...')
        splitter = self.conf['dataset.split']
        val_ratio = self.conf['dataset.val_ratio']
        test_split= 0.2
        df=pd.read_csv(os.path.join(self.data_path,self.dataset)+'/data.txt',header=None,names=['userId', 'timestamp', 'itemId','sessionId'])
        print(df['userId'].nunique())
        print(df['itemId'].nunique())
        print(df['itemId'].max(),df['itemId'].min())
        endtime = df.groupby('sessionId', sort=False).timestamp.max()
        endtime = endtime.sort_values()
        num_tests = int(len(endtime) * test_split)
        test_session_ids = endtime.index[-num_tests:]
        df_train = df[~df.sessionId.isin(test_session_ids)]
        df_test = df[df.sessionId.isin(test_session_ids)].reset_index(drop=True)

        ## remap index
        df_test = df_test[df_test.itemId.isin(df_train.itemId.unique())]
        df_test = self.filter_short_sessions(df_test)

        train_itemId_new, uniques = pd.factorize(df_train.itemId)
        df_train = df_train.assign(itemId=train_itemId_new)
        oid2nid = {oid: i for i, oid in enumerate(uniques)}
        test_itemId_new = df_test.itemId.map(oid2nid)
        df_test = df_test.assign(itemId=test_itemId_new)
        df_train['userId']+=1
        df_train['itemId']+=1
        df_test['userId']+=1
        df_test['itemId']+=1

        self._agg_all_seq(df_train)


        print(df_train['userId'].min(),df_train['userId'].max())
        print(df_train['itemId'].max(),df_train['itemId'].min())
        print(df_test['itemId'].max(),df_test['itemId'].min())

        # split 
        df_test=df_test.reset_index(drop=True)
        df_val= df_test.sample(frac=val_ratio)
        part_test=df_test[~df_test.index.isin(df_val.index)]
        
        with open(os.path.join(self.data_path,self.dataset)+'/train.pkl','wb') as f:
            pickle.dump(self._agg_df(df_train),f)
        
        with open(os.path.join(self.data_path,self.dataset)+'/val.pkl','wb') as f:
            pickle.dump(self._agg_df(df_val),f)

        with open(os.path.join(self.data_path,self.dataset)+'/test.pkl','wb') as f:
            pickle.dump(self._agg_df(part_test),f)
        
        with open(os.path.join(self.data_path,self.dataset)+'/all_test.pkl','wb') as f:
            pickle.dump(self._agg_df(df_test),f)

    def _read_raw_data(self,topK=40000):
        data_home='./raw_data/lastfm-1K'
        saved_path='./dataset'
        csv_file=data_home+'/userid-timestamp-artid-artname-traid-traname.tsv'
        print(f'read raw data from {csv_file}')
        df = pd.read_csv(
            csv_file,
            sep='\t',
            header=None,
            names=['userId', 'timestamp', 'itemId','artname','traid','traname'],
            usecols=['userId', 'timestamp', 'itemId'],
            parse_dates=['timestamp'],
            infer_datetime_format=True,
        )
        print('start preprocessing')
        df = df.dropna()
        df = self._update_id(df,'userId')
        df = self._update_id(df, 'itemId')
        df = df.sort_values(['userId', 'timestamp'])

        df = self._group_sessions(df)
        df = self.remove_immediate_repeats(df)
        df = self.truncate_long_sessions(df, is_sorted=True)
        df = self.keep_top_n_items(df, topK)
        df = self.filter_until_all_long_and_freq(df)
        if not os.path.exists(f'{saved_path}/lastfm'):
            os.mkdir(f'{saved_path}/lastfm')
        df.to_csv(f'{saved_path}/lastfm/'+'data.txt',sep=',',header=None,index=False)
        # df_train, df_test = train_test_split(df, test_split=0.2)
        # save_dataset(dataset_dir, df_train, df_test)

    def _get_user_profile(self):
        
        data_home='./raw_data/lastfm-1K'
        saved_path='./dataset'
        csv_file=data_home+'/userid-timestamp-artid-artname-traid-traname.tsv'
        profile_file=data_home+'/userid-profile.tsv'
        print(f'read raw data from {csv_file}')
        df = pd.read_csv(
            csv_file,
            sep='\t',
            header=None,
            names=['userId', 'timestamp', 'itemId','artname','traid','traname'],
            usecols=['userId', 'timestamp', 'itemId'],
            parse_dates=['timestamp'],
            infer_datetime_format=True,
        )
        user_df = pd.read_csv(
            profile_file,
            sep='\t'
            #names=['userId', 'timestamp', 'itemId','artname','traid','traname'],
            #usecols=['userId', 'timestamp', 'itemId'],
           # parse_dates=['timestamp'],
           # infer_datetime_format=True,
        )
        print('get user_profile')
        df = df.dropna()
        uids=df['userId'].unique().tolist()
        user_df[user_df['#id'].isin(uids)].to_csv(saved_path+'/lastfm/user_profile.csv',index=False)
        print('save to user_profile.csv')
        
if __name__=='__main__':
    conf={}
    lp=LastFM_Process(conf)
    lp._read_raw_data()
    lp._split_data()