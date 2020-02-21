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
# from tigerforecast.data.ealstm_regional_modeling import main

import numpy as onp
import random as rnd
import sys
from tqdm import tqdm
# from tigerforecast.data.ealstm_regional_modeling.papercode.utils import get_basin_list
# from tigerforecast.data.ealstm_regional_modeling.papercode.datautils import rescale_features
import gc
from tigerforecast.utils.dynamic.DynamicEval import *
from tigerforecast.methods.boosting.SimpleBoostHet import *
from tigerforecast.batch.camels_dataloader import *

# all_sites = get_basin_list()
REG = 0.0
# NUM_SITES = len(all_sites)
TRAINING_STEPS = 1000000
BATCH_SIZE = 256
SEQUENCE_LENGTH = 270
HIDDEN_DIM = 256
DP_RATE = 0.4
LR_ar = 0.0
LR_LSTM = 0.001
INDICES_TO_KEEP = [15, 16, 17, 18, 19, 20, 21, 22, 26, 27, 28, 29, 30, 31, 32, 34, 35, 37, 38, 39, 40, 41, 48, 49, 50, 56, 58, 60, 61, 62, 63, 64]
INPUT_DIM = len(INDICES_TO_KEEP)
TRAIN_SEQ_TO_SEQ = False
# optim = OGD(loss=batched_mse, learning_rate=0.1)
hyperparams = {'reg':0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'eps': 1e-8, 'max_norm':True}
optim_lstm = Adam(loss=batched_mse, learning_rate=LR_LSTM, include_x_loss=False, hyperparameters=hyperparams)

if TRAIN_SEQ_TO_SEQ:
        method_LSTM = tigerforecast.method("Seq2SeqLSTM")
else:
        method_LSTM = tigerforecast.method("Seq2ValLSTM")
method_LSTM.initialize(n=INPUT_DIM, m=1, l=SEQUENCE_LENGTH, h=HIDDEN_DIM, optimizer=optim_lstm, dp_rate=DP_RATE)
# method_LSTM.load('trained_model/trained_CAMELS.pkl')
method_LSTM.load('trained_CAMEL_torchlessdataloader_11.pkl')
# optim_ar = Adam(loss=batched_mse, learning_rate=LR, include_x_loss=False, hyperparameters=hyperparams)
optim_ar = OGD(loss=batched_mse, learning_rate=LR_ar)
method_ar = tigerforecast.method("ARStateless")
method_ar.initialize(n=INPUT_DIM, m=1, l=SEQUENCE_LENGTH, optimizer=optim_ar)

results_boosted = []
pred_boosted = []

SCALER = {
            'input_means': np.array([3.17563234, 372.01003929, 17.31934062, 3.97393362, 924.98004197]),
            'input_stds': np.array([6.94344737, 131.63560881, 10.86689718, 10.3940032, 629.44576432]),
            'output_mean': np.array([1.49996196]),
            'output_std': np.array([3.62443672])
        }

def usgs_eval_LSTM(method_LSTM, usgs_val, batch_size=None, dynamic=False):
        ### We will do this with the NE metric
        yhats_LSTM, ys = [], []
        # for x_y in usgs_val:
        for data, targets in usgs_val.sequential_batches(batch_size=batch_size):
                # data, targets = x_y
                # data = data.numpy()
                # targets = targets.numpy()
                data = np.array(data)
                targets = np.array(targets)
                y_pred_lstm = method_LSTM.predict(data)
                y_pred_lstm = SCALER['output_std'] * y_pred_lstm + SCALER['output_mean']
                y_pred_lstm = np.where(y_pred_lstm < 0.0, 0.0, y_pred_lstm)
                yhats_LSTM.append(y_pred_lstm[:,-1].copy())
                ys.append(targets[:,-1].copy())
                method_LSTM.update(targets[:,-1].copy())
        # yhats_boosted = rescale_features(onp.array(yhats_boosted), variable='output')
        # yhats_shapes = [x.shape for x in yhats_boosted]
        # print(yhats_shapes)
        # yhats_flat = [x.flatten() for x in yhats]
        yhats_LSTM = np.array(onp.hstack(yhats_LSTM))
        gc.collect()
        return np.array(yhats_LSTM), np.array(np.concatenate(ys).ravel())

def usgs_eval_boosted(yhats_LSTM, method_ar, usgs_val, batch_size=None, dynamic=False):
        ### We will do this with the NE metric
        yhats_boosted, ys = [], [] 
        #for data, targets in usgs_val.sequential_batches(site_idx=site_idx, batch_size=batch_size):
        # for x_y in usgs_val:
        cnt = 0
        # print("yhats_LSTM.shape = " + str(yhats_LSTM.shape))
        for data, targets in usgs_val.sequential_batches(batch_size=batch_size):
                data = np.array(data)
                targets = np.array(targets)
                y_pred_ar = method_ar.predict(data)
                # print("y_pred_ar.shape = " + str(y_pred_ar.shape))
                # print("y_pred_lstm.shape = " + str(y_pred_lstm.shape))
                # print("(targets - y_pred_lstm).shape = " + str((targets - y_pred_lstm).shape))
                y_pred_ar = np.array([a.flatten() for a in y_pred_ar])
                # if cnt % 250 == 0:
                #         print("y_pred_lstm[:,-1] = " + str(y_pred_lstm[:10,-1]))
                #         print("y_pred_ar[:,-1] = " + str(y_pred_ar[:10,-1]))
                # targets_exp = np.expand_dims(targets[:,-1], axis=-1)
                # if dynamic:
                #        eval_method.update(targets_exp)
                method_ar.update(np.expand_dims(targets - yhats_LSTM[cnt], -1))
                # yhats_boosted.append(y_pred_lstm[:,-1].copy() + y_pred_ar[:,-1].copy())
                yhats_boosted.append(yhats_LSTM[cnt] + y_pred_ar[:,-1].copy())
                ys.append(targets[:,-1].copy())
                cnt += 1
        yhats_boosted = np.array(onp.hstack(yhats_boosted))
        gc.collect()
        return np.array(yhats_boosted), np.array(np.concatenate(ys).ravel())

# user_cfg, run_cfg, db_path, means, stds = main.make_eval_usercfg_runcfg_dbpath_means_stds()
losses = []
losses_no_outliers = []
losses_yes_outliers = []
nses = []
nses_no_outliers = []
LR_lstm = [0.0, 0.1, 0.01, 0.001, 0.0001]
# LR_lstm = [0.0, 1.0, 0.9, 0.8, 0.7, 0.6]
tigerforecast_dir = get_tigerforecast_dir()
BASIN_PATH = os.path.join(tigerforecast_dir, 'data/usgs_flood/basin_list.txt')
all_sites = get_basin_list(BASIN_PATH)
site0_yhats_LSTM = []
site0_yhsts_boosted = []
site0_ys = []

for lr in LR_lstm:
       print("lr = " + str(lr))
       # print("optim = OGD")
       M1 = tigerforecast.method("Seq2ValLSTM")
       M2 = tigerforecast.method("Seq2ValLSTM")
       opt1 = OGD(loss=batched_mse, learning_rate=lr)
       opt2 = OGD(loss=batched_mse, learning_rate=lr)
       M1.initialize(n=INPUT_DIM, m=1, l=SEQUENCE_LENGTH, h=HIDDEN_DIM, optimizer=opt1, dp_rate=DP_RATE)
       M2.initialize(n=INPUT_DIM, m=1, l=SEQUENCE_LENGTH, h=HIDDEN_DIM, optimizer=opt2, dp_rate=DP_RATE)
       M1.load('trained_CAMEL_torchlessdataloader_11.pkl')
       # M2.load('trained_CAMEL_torchlessdataloader_11.pkl')
       method_boosted = SimpleBoostHet()
       method_boosted.initialize([M1, M2], loss=batched_mse, reg=REG)
       for i, site in enumerate(all_sites[:20]):
              if i % 10 == 0:
                      print("i = " + str(i))
              usgs_val = CamelsTXT(basin=site, concat_static=True)
              # usgs_val = main.make_eval_data_loader(site, user_cfg, run_cfg, db_path, means, stds)
              yhats, ys = usgs_eval_LSTM(method_boosted, usgs_val, batch_size=1024, dynamic=False)
              # yhats, ys = usgs_eval_boosted(yhats_LSTM, method_ar, usgs_val, batch_size=1, dynamic=False)
                                                                                                                          
              loss = ((ys-yhats)**2).mean()
              # print("loss = " + str(loss))
              ys_mean = ys.mean()
              nse = (1 - ((ys - yhats)**2).sum()/((ys - ys_mean)**2).sum())
              # print("nse = " + str(nse))
              if np.abs(nse) < 2:
                      nses_no_outliers.append(nse)
                      losses_no_outliers.append(loss)
              else:
                      losses_yes_outliers.append(loss)
              if not np.isnan(nse):
                      nses.append(nse)
                      losses.append(loss)
              # method_ar = method_ar_updated
              gc.collect()
       avg_loss = np.array(losses).mean()
       avg_loss_no_outliers = np.array(losses_no_outliers).mean()
       avg_nse = np.array(nses).mean()
       avg_nse_no_outliers = np.array(nses_no_outliers).mean()
       # print("prior_wt = " + str(prior_wt))
       print("avg_loss = " + str(avg_loss))
       print("avg_loss_no_outliers = " + str(avg_loss_no_outliers))
       # print("losses_yes_outliers = " + str(losses_yes_outliers))
       print("avg_nse = " + str(avg_nse))
       print("avg_nse_no_outliers = " + str(avg_nse_no_outliers))         
