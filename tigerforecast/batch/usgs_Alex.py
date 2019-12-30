# data loader for USGS flood data

import pandas as pd
import numpy as np
import random
import os
import itertools
import ast




# order of columns:
# 0 = gauge
# 1 - 60 = static features
# 60 - 64 = seq_features
# 65    b = target

# Notes: features and target are already normalized by subtracting mean and dividing by stds

FACTORIZE = ['gauge_id', 'gauge_name', 'high_prec_timing', 'low_prec_timing', 'dom_land_cover', 'geol_1st_class', 'geol_2nd_class']
NUM_FEATURES = 66

class USGSDataLoader:
    def __init__(self, mode=None, seq_length=270, gaugeID_to_idx=None, normalize_source=None):
        self.seq_length = seq_length
        self.target_lag = 1

        # load and group data
        # self.df = pd.read_csv(path)
        static_features_path = '../data/usgs_flood/attributes.csv'
        if mode == 'train':
            seq_features_path = '../data/usgs_flood/train_data.csv'
        elif mode == 'val':
            seq_features_path = '../data/usgs_flood/val_data.csv'

        self.df_static = pd.read_csv(static_features_path, converters={'gauge_id': lambda x: str(x)})
        for col in FACTORIZE:
            if col == 'gauge_id':
                self.df_static[col], self.unique_gauge_ids = pd.factorize(self.df_static[col])
                self.gaugeID_to_idx = {x:i for i,x in enumerate(self.unique_gauge_ids)}
            else:
                self.df_static[col] = pd.factorize(self.df_static[col])[0]

        # print(self.gaugeID_to_idx)

        self.features_static = self.df_static.columns

        if mode == 'eval':
            for col in self.features_static:
                # print("col = " + str(col))
                if col not in FACTORIZE:
                    # print("not in factorize")
                    mean = self.df_static[col].mean()
                    std = self.df_static[col].std()
                    self.df_static[col] -= mean
                    self.df_static[col] /= std + 1e-4
                # mean = self.np_data[self.gaugeID_to_idx[key]][0][:,feature].mean()
                # std = self.np_data[self.gaugeID_to_idx[key]][0][:,feature].std()
                # self.np_data[self.gaugeID_to_idx[key]][0][:,feature] -= mean
                # self.np_data[self.gaugeID_to_idx[key]][0][:,feature] /= std + 1e-4

        self.df_seq = pd.read_csv(seq_features_path)
        self.df_seq = self.df_seq.drop(self.df_seq.columns[-2],axis=1) # drop q_std

        # self.features_seq = self.df_seq.columns
        self.groups = self.df_seq.groupby(by='basin_str')
        self.gauge_keys = list(self.groups.groups.keys())
        self.gauge_keys = [key[2:-1] for key in self.gauge_keys]
        self.gauge_keys.sort()

        # if metrics_path:
        #    self.df_metrics = pd.read_csv(metrics_path)
        # self.groups_metrics = self.df_metrics.groupby(by='__site_id')
        # self.gauge_keys_metrics = list(self.groups_metrics.groups.keys())
        # self.gauge_keys_metrics.sort()

        # site_keys: index -> site_id
        # site_idx : site_id -> index
        
        if gaugeID_to_idx:
            self.gaugeID_to_idx = gaugeID_to_idx
            self.gauge_keys = [x for x in self.gauge_keys if x in gaugeID_to_idx]
 
        # turn group data into numpy arrays
        cols = []
        self.np_data = dict()
        self.data_length = dict()

        for key, df in self.groups:
            key = key[2:-1]
            if key not in self.gaugeID_to_idx:
                continue

            # data = df[FEATURES].to_numpy()
            # print(self.df_static['gauge_id'][0])
            # print(type(self.df_static['gauge_id'][0]))

            data_seq = df.to_numpy()[:,1:-1] # drop first col (index) and last col (basin_str) since it's redundant with gauge_id
            # print(data_seq[0])
            # print(data_seq[1])
            data_static = self.df_static.loc[self.df_static['gauge_id'] == self.gaugeID_to_idx[key]].to_numpy()
            data = [None for i in range(data_seq.shape[0])]

            # print("key = " + str(key))
            # print("key.decode() = " + key[2:-1])
            # print(data_static)
            for i in range(data_seq.shape[0]):
                data[i] = np.hstack((data_static[0], data_seq[i,:]))
            # print(data[0])

            # df_metrics = self.groups_metrics.get_group(key)
            
            # last index 1 locates target discharge_mean; see FEATURES
            data = np.array(data)
            targets = data[:,-1].copy()
            # mask the current target
            data[self.target_lag:,-1] = data[:-self.target_lag,-1]

            # compress site_ids
            data[:,0] = self.gaugeID_to_idx[key]
            '''if metrics_path:
                data_metrics = self.df_metrics.loc[self.df_metrics['__site_id'] == key].to_numpy()
                data_metrics[:,0] = self.gaugeID_to_idx[key]
                data_metric_row = data_metrics[0]
                data_metric_row = data_metric_row[2:]
                data_metric_row[1] = np.around(np.array(ast.literal_eval(data_metric_row[1])), 3)
                data_metric_row[2] = np.around(np.array(ast.literal_eval(data_metric_row[2])), 3)'''

            # print(data[0])
            # print(targets[0])
            # print("len(data_static[0]) = " + str(len(data_static[0])))
            # print("len(data[0]) = " + str(len(data[0])))

            self.np_data[self.gaugeID_to_idx[key]] = (data, targets)
            self.data_length[self.gaugeID_to_idx[key]] = len(data)

        # normalize from your own per-site statistics if None, or from reference (train) set 
        
        # self.normalize_moments = dict()
        # self.normalize_source = normalize_source
        

    def featurize(self, site_idx, t):
        # grab (data, target) sequence from a slice of a time series
        
        # disallow indices without fully specified features
        if t < self.seq_length + self.target_lag - 1:
            raise KeyError

        all_data, all_targets = self.np_data[site_idx]

        data = all_data[t-self.seq_length+1:t+1, :]
        target = all_targets[t-self.seq_length+1:t+1]

        return data, target

    def get_data_metrics(self, site_idx):
        return self.np_data[site_idx][2]
        '''
        row = self.np_data[site_idx][2][0][2:]
        if not isinstance(row[1], np.ndarray):
            row[1] = np.around(np.array(ast.literal_eval(row[1])), 3)
            row[2] = np.around(np.array(ast.literal_eval(row[2])), 3)
        return row'''

    def random_site_idx(self):
        # generates a random site index
        return self.gaugeID_to_idx[ random.choice(self.gauge_keys) ]

    def random_valid_time(self, site_idx):
        # generates a random time in a given site
        return random.randint(self.seq_length + self.target_lag - 1,
            self.data_length[site_idx] - 1)

    def random_batches(self, batch_size, num_batches=None):
        # generator for random batches
        # default num_batches: infinity

        batch_data = np.zeros([batch_size, self.seq_length, NUM_FEATURES])
        batch_targets = np.zeros([batch_size, self.seq_length])

        for _ in itertools.repeat(None, num_batches):
            for i in range(batch_size):
                rand_idx = self.random_site_idx()
                rand_t = self.random_valid_time(rand_idx)
                # print(rand_idx, rand_t)
                data, target = self.featurize(rand_idx, rand_t)
                batch_data[i] = data
                batch_targets[i] = target
            yield batch_data, batch_targets

    def sequential_batches(self, site_idx, batch_size, num_batches=None, skip_first=0):
        # generator for sequential batches

        batch_data = np.zeros([batch_size, self.seq_length, NUM_FEATURES])
        batch_targets = np.zeros([batch_size, self.seq_length])

        # for now, truncate the end
        first = self.seq_length + self.target_lag - 1 + skip_first
        if not num_batches:
            num_batches = (self.data_length[site_idx] - first) // batch_size

        for k in range(num_batches):
            for i in range(batch_size):
                t = first + k*batch_size + i
                # print(site_idx, t)
                data, target = self.featurize(site_idx, t)
                batch_data[i] = data
                batch_targets[i] = target
            yield batch_data, batch_targets

usgs_train = USGSDataLoader(mode='train')

usgs_val = USGSDataLoader(mode='val', gaugeID_to_idx=usgs_train.gaugeID_to_idx)

# # get site 0, time 123
data, target = usgs_train.featurize(1, 300)
print(data.shape, target.shape)

# # probe randomly into the dataset
for data, targets in usgs_train.random_batches(batch_size=8, num_batches=5):
     print(data.shape, targets.shape, data[:,0,1])

# # probe sequentially to cover a single site's time series
for data, targets in usgs_val.sequential_batches(site_idx=1, batch_size=1, num_batches=10):
     print(data.shape, targets.shape, data[:,0:5,64])

# print(data, target)
