import tigerforecast
import jax
import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
import pickle
from tigerforecast.utils import generate_key
from tigerforecast.batch.usgs_Alex import *

from tigerforecast.utils.optimizers import *
from tigerforecast.utils.optimizers.losses import *
from tigerforecast.flood_prediction.ealstm_regional_modeling import main


import numpy as onp
import random as rnd
import sys
from tqdm import tqdm
from tigerforecast.data.ealstm_regional_modeling.papercode.utils import get_basin_list
from tigerforecast.data.ealstm_regional_modeling.papercode.datautils import rescale_features
import gc

TRAINING_STEPS = 1000000
BATCH_SIZE = 256
SEQUENCE_LENGTH = 270
HIDDEN_DIM = 256
EMBEDDING_DIM = 10
DP_RATE = 0.4
LR = 0.001
OFFSET = 0
INDICES_TO_KEEP = [15, 16, 17, 18, 19, 20, 21, 22, 26, 27, 28, 29, 30, 31, 32, 34, 35, 37, 38, 39, 40, 41, 48, 49, 50, 56, 58, 60, 61, 62, 63, 64]
INPUT_DIM = len(INDICES_TO_KEEP)
# path_csv = os.path.join(tigerforecast_dir, 'data/unemployment.csv')
DATA_PATH = '../data/usgs_flood/usgs_{}.csv'
TRAIN_SEQ_TO_SEQ = False
# MODEL_PATH = 'trained_CAMEL_5000.pkl'
# optim = OGD(loss=batched_mse, learning_rate=0.1)
hyperparams = {'reg':0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'eps': 1e-8, 'max_norm':True}
optim = Adam(loss=batched_mse, learning_rate=LR, include_x_loss=False, hyperparameters=hyperparams)

# usgs_train = USGSDataLoader(mode='train')
#usgs_val = USGSDataLoader(mode='val', seq_length=SEQUENCE_LENGTH, gaugeID_to_idx=usgs_train.gaugeID_to_idx)



if TRAIN_SEQ_TO_SEQ:
        method_LSTM = tigerforecast.method("Seq2SeqLSTM")
else:
        method_LSTM = tigerforecast.method("Seq2ValLSTM")
method_LSTM.initialize(n=INPUT_DIM, m=1, l=SEQUENCE_LENGTH, h=HIDDEN_DIM, optimizer=optim, dp_rate=DP_RATE)
# method_LSTM.load('trained_CAMEL_newdp_0.1_190000.pkl')
results_LSTM = []
pred_LSTM = []

def usgs_eval(method, site_idx, usgs_val, batch_size=None, dynamic=False, path=None):
        ### We will do this with the NE metric
        #optim = OGD(loss=batched_mse, learning_rate=LR)
        # optim = Adam(loss=batched_mse, learning_rate=LR, include_x_loss=False, hyperparameters=hyperparams)
        if dynamic:
                optim = DynamicEval(optim, prior_weight=0.0)
        eval_method = tigerforecast.method("Seq2ValLSTM")
        eval_method.initialize(n=INPUT_DIM, m=1, l=SEQUENCE_LENGTH, h=HIDDEN_DIM, dp_rate=0.0)
        if not path:
                eval_method.params = method.copy()
        else:
                eval_method.load(path)
        if dynamic:
                dynamic_optim.set_prior(eval_method.params)

        yhats, ys = [], [] 
        #for data, targets in usgs_val.sequential_batches(site_idx=site_idx, batch_size=batch_size):
        for x_y in usgs_val:
                data, targets = x_y
                data = data.numpy()
                targets = targets.numpy()
                y_pred = eval_method.predict(data)
                targets_exp = np.expand_dims(targets[:,-1], axis=-1)
                if dynamic:
                        eval_method.update(targets_exp)
                yhats.append(y_pred[:,-1].copy())
                ys.append(targets[:,-1].copy())
        # print(yhats[:50])
        yhats = rescale_features(onp.array(yhats), variable='output')
        # yhats[yhats < 0] = 0
        # print(type(yhats))
        # yhats_shapes = [y.shape for y in yhats]
        # print(yhats_shapes[:50])
        # ys_shapes = [y.shape for y in ys]
        # print(ys_shapes[:50])
        yhats = np.array(onp.hstack(yhats))
        yhats = np.where(yhats < 0.0, 0.0, yhats)
        # print(type(yhats))
        return yhats, np.array(np.concatenate(ys).ravel())
best_index = -1
best_loss = 1000
best_nse = -1
best_index_no_outliers = -1
best_loss_no_outliers = 1000
best_nse_no_outliers = -1
# all_sites = usgs_train.get_all_sites()
all_sites = get_basin_list()
# sites_100 = rnd.sample(all_sites, 100)
usgs_train = main.get_data_loader('train')
# for i, (data, targets) in enumerate( usgs_train.random_batches(batch_size=BATCH_SIZE, num_batches=TRAINING_STEPS) ):
user_cfg, run_cfg, db_path, means, stds = main.make_eval_usercfg_runcfg_dbpath_means_stds()

learning_rates = {11: 5e-4, 21: 1e-4}
for epoch in range(1,31):
       for i, x_y_z in enumerate(usgs_train):
               if epoch in learning_rates.keys():
                       method_LSTM.set_learning_rate(learning_rates[epoch])
               data, targets, q_stds = x_y_z
               data = data.numpy()
               targets = targets.numpy()
               y_pred_LSTM = method_LSTM.predict(data)
               if TRAIN_SEQ_TO_SEQ:
                       targets_exp = np.expand_dims(targets, axis=-1)
               else:
                       targets_exp = np.expand_dims(targets[:,-1], axis=-1)            
                       loss = float(batched_mse(y_pred_LSTM, targets_exp))
                       # print("i = " + str(i))
                       # print("loss = " + str(loss))
               results_LSTM.append(loss)
               method_LSTM.update(targets_exp) 
               if i % 400 == 0:
                       print("i = " + str(i))
               if i%6800 == 0 and i > 0:
                       print('epoch = ' + str(epoch))
                       print('Step %i: loss=%f' % (i,results_LSTM[-1]) )
                       method_LSTM.save('trained_CAMEL_newdataloader_' + str(epoch) + '.pkl')
                       losses = []
                       losses_no_outliers = []
                       nses = []
                       nses_no_outliers = []

                       for site in all_sites:
                               # print("Doing Site ", site)
                               usgs_val = main.make_eval_data_loader(site, user_cfg, run_cfg, db_path, means, stds)
                               yhats, ys = usgs_eval(method_LSTM, site, usgs_val, batch_size=1024, dynamic=False)
                               loss = ((ys-yhats)**2).mean()
                               # print("loss = " + str(loss))
                               ys_mean = ys.mean()
                               nse = (1 - ((ys - yhats)**2).sum()/((ys - ys_mean)**2).sum())
                               # print("nse = " + str(nse))
                               if np.abs(nse) < 2:
                                       nses_no_outliers.append(nse)
                                       losses_no_outliers.append(loss)
                               if not np.isnan(nse):
                                       nses.append(nse)
                                       losses.append(loss)
                                       gc.collect()
                               avg_loss = np.array(losses).mean()
                       avg_loss_no_outliers = np.array(losses_no_outliers).mean()
                       avg_nse = np.array(nses).mean()
                       avg_nse_no_outliers = np.array(nses_no_outliers).mean()
                       print("avg_loss = " + str(avg_loss))
                       print("avg_loss_no_outliers = " + str(avg_loss_no_outliers))
                       print("avg_nse = " + str(avg_nse))
                       print("avg_nse_no_outliers = " + str(avg_nse_no_outliers))
                       if avg_loss < best_loss:
                               best_loss = avg_loss
                               best_index = i
                               best_nse= avg_nse               

                               
