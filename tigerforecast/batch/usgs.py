# data loader for USGS flood data

import pandas as pd
import numpy as np
import random
import os
import itertools
import ast

FEATURES = ['__site_id',
    'sequence:USGS:discharge_mean',
    'sequence:AQUA_VI:NDVI',
    'sequence:GLDAS21:Tair_f_inst',
    'sequence:GSMAP_MERGED:hourlyPrecipRate',
    'static:drain_area_log2',
    'train:site:mean:USGS:discharge_mean',
    'train:site:std:USGS:discharge_mean']

FEATURES_METRICS = ['__site_id', 'metrics:return_period_values', 'metrics:return_periods']

NORMALIZE_INDICES = [2,3]

class USGSDataLoader:
    def __init__(self, path, metrics_path=None, seq_length=61, site_idx=None, normalize_source=None):
        self.seq_length = seq_length
        self.target_lag = 1

        # load and group data
        self.df = pd.read_csv(path)
        self.groups = self.df.groupby(by='__site_id')
        self.site_keys = list(self.groups.groups.keys())
        self.site_keys.sort()

        self.df_metrics = pd.read_csv(metrics_path)
        # self.groups_metrics = self.df_metrics.groupby(by='__site_id')
        # self.site_keys_metrics = list(self.groups_metrics.groups.keys())
        # self.site_keys_metrics.sort()

        # site_keys: index -> site_id
        # site_idx : site_id -> index
        if site_idx:
            self.site_idx = site_idx
            self.site_keys = [x for x in self.site_keys if x in site_idx]
        else:
            self.site_idx = {x:i for i,x in enumerate(self.site_keys)}
 
        # turn group data into numpy arrays
        cols = []
        self.np_data = dict()
        self.data_length = dict()

        for key, df in self.groups:
            if key not in self.site_idx:
                continue
            data = df[FEATURES].to_numpy()

            # df_metrics = self.groups_metrics.get_group(key)
            data_metrics = self.df_metrics.loc[self.df_metrics['__site_id'] == key].to_numpy()
            # last index 1 locates target discharge_mean; see FEATURES
            targets = data[:,1].copy()
            # mask the current target
            data[self.target_lag:,1] = data[:-self.target_lag,1]

            # compress site_ids
            data[:,0] = self.site_idx[key]
            data_metrics[:,0] = self.site_idx[key]
            data_metric_row = data_metrics[0]
            data_metric_row = data_metric_row[2:]
            data_metric_row[1] = np.around(np.array(ast.literal_eval(data_metric_row[1])), 3)
            data_metric_row[2] = np.around(np.array(ast.literal_eval(data_metric_row[2])), 3)

            self.np_data[self.site_idx[key]] = (data, targets, data_metric_row)
            self.data_length[self.site_idx[key]] = len(data)

        # normalize from your own per-site statistics if None, or from reference (train) set 
        self.normalize_moments = dict()
        self.normalize_source = normalize_source
        for key, _ in self.groups:
            if key not in self.site_idx:
                continue
            for feature in NORMALIZE_INDICES:
                if normalize_source:
                    mean, std = normalize_source.normalize_moments[(self.site_idx[key], feature)]
                else:
                    mean = self.np_data[self.site_idx[key]][0][:,feature].mean()
                    std = self.np_data[self.site_idx[key]][0][:,feature].std()
                    self.normalize_moments[(self.site_idx[key], feature)] = mean, std
                self.np_data[self.site_idx[key]][0][:,feature] -= mean
                self.np_data[self.site_idx[key]][0][:,feature] /= std + 1e-4

    def featurize(self, site_idx, t):
        # grab (data, target) sequence from a slice of a time series
        
        # disallow indices without fully specified features
        if t < self.seq_length + self.target_lag - 1:
            raise KeyError

        all_data, all_targets, _ = self.np_data[site_idx]

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
        return self.site_idx[ random.choice(self.site_keys) ]

    def random_valid_time(self, site_idx):
        # generates a random time in a given site
        return random.randint(self.seq_length + self.target_lag - 1,
            self.data_length[site_idx] - 1)

    def random_batches(self, batch_size, num_batches=None):
        # generator for random batches
        # default num_batches: infinity

        batch_data = np.zeros([batch_size, self.seq_length, len(FEATURES)])
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

        batch_data = np.zeros([batch_size, self.seq_length, len(FEATURES)])
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

# usgs_train = USGSDataLoader(DATA_PATH.format('train'))
# usgs_val = USGSDataLoader(DATA_PATH.format('val'), site_idx=usgs_train.site_idx)

# # get site 0, time 123
# data, target = usgs_train.featurize(0, 123)
# print(data.shape, target.shape)

# # probe randomly into the dataset
# for data, targets in usgs_train.random_batches(batch_size=8, num_batches=5):
#     print(data.shape, targets.shape, data[:,0,1])

# # probe sequentially to cover a single site's time series
# for data, targets in usgs_val.sequential_batches(site_idx=0, batch_size=1, num_batches=10):
#     print(data.shape, targets.shape, data[:,0:5,1])

# print(data, target)
