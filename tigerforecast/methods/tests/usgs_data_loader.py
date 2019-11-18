# data loader for USGS flood data

import pandas as pd
import numpy as np
import random
import os
import itertools

FEATURES = ['__site_id',
    'label:USGS:discharge_mean',
    'sequence:AQUA_VI:NDVI',
    'sequence:GLDAS21:Tair_f_inst',
    'sequence:GSMAP_MERGED:hourlyPrecipRate',
    'sequence:USGS:discharge_mean',
    'static:drain_area_log2',
    'train:site:mean:USGS:discharge_mean',
    'train:site:std:USGS:discharge_mean']

class USGSDataLoader:
    def __init__(self, path, seq_length=61, site_idx=None): 
        self.seq_length = seq_length
        self.target_lag = 1

        # load and group data
        self.df = pd.read_csv(path)
        self.groups = self.df.groupby(by='__site_id')
        self.site_keys = list(self.groups.groups.keys())
        self.site_keys.sort()

        # site_keys: index -> site_id
        # site_idx : site_id -> index
        if site_idx:
            self.site_idx = site_idx
        else:
            self.site_idx = {x:i for i,x in enumerate(self.site_keys)}
 
        # turn group data into numpy arrays
        cols = []
        self.np_data = dict()
        self.data_length = dict()
        for key, df in self.groups:
            data = df[FEATURES].to_numpy()
            # last index 1 locates target discharge_mean; see FEATURES
            targets = data[:,1]
            # mask the current target
            data[self.target_lag:,1] = data[:-self.target_lag,1]

            self.np_data[self.site_idx[key]] = (data, targets)
            self.data_length[self.site_idx[key]] = len(data)

    def featurize(self, site_idx, t):
        # grab (data, target) sequence from a slice of a time series
        
        # disallow indices without fully specified features
        if t < self.seq_length + self.target_lag - 1:
            raise KeyError

        all_data, all_targets = self.np_data[site_idx]

        data = all_data[t-self.seq_length+1:t+1, :]
        target = all_targets[t-self.seq_length+1:t+1]

        return data, target

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