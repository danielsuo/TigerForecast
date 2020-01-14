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

import numpy as onp
import random as rnd
# lr = 1e-3
'''
Step 1100: loss=1.119998
Step 1200: loss=1.075929
Step 1300: loss=1.106990
Step 1400: loss=1.262505
Step 1500: loss=0.801956
Step 1600: loss=0.780336
Step 1700: loss=1.247742
Step 1800: loss=1.151220
Step 1900: loss=1.242591
Step 2000: loss=1.310133
Eval: loss=9.082329
NSE: -0.844390
'''
# lr = 1e-2
'''
Step 1100: loss=0.909760
Step 1200: loss=1.201821
Step 1300: loss=1.281517
Step 1400: loss=0.499185
Step 1500: loss=0.767103
Step 1600: loss=0.900516
Step 1700: loss=0.739443
Step 1800: loss=1.022097
Step 1900: loss=0.600303
Step 2000: loss=1.384120
Eval: loss=8.140976
NSE: -0.653225
'''

# lr = 1e-1


TRAINING_STEPS = 100000
BATCH_SIZE = 256
SEQUENCE_LENGTH = 270
HIDDEN_DIM = 256
EMBEDDING_DIM = 10
DP_RATE = 0.4
LR = 0.01
INDICES_TO_KEEP = [15, 16, 17, 18, 19, 20, 21, 22, 26, 27, 28, 29, 30, 31, 32, 34, 35, 37, 38, 39, 40, 41, 48, 49, 50, 56, 58, 60, 61, 62, 63, 64, 65]
INPUT_DIM = len(INDICES_TO_KEEP)
# path_csv = os.path.join(tigerforecast_dir, 'data/unemployment.csv')
DATA_PATH = '../data/usgs_flood/usgs_{}.csv'
TRAIN_SEQ_TO_SEQ = False
MODEL_PATH = 'trained_CAMEL_5000.pkl'
# optim = OGD(loss=batched_mse, learning_rate=0.1)
hyperparams = {'reg':0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'eps': 1e-8, 'max_norm':True}
optim = Adam(loss=batched_mse, learning_rate=LR, include_x_loss=False, hyperparameters=hyperparams)

usgs_train = USGSDataLoader(mode='train')
usgs_val = USGSDataLoader(mode='val', seq_length=SEQUENCE_LENGTH, gaugeID_to_idx=usgs_train.gaugeID_to_idx)

if TRAIN_SEQ_TO_SEQ:
	method_LSTM = tigerforecast.method("Seq2SeqLSTM")
else:
	method_LSTM = tigerforecast.method("Seq2ValLSTM")
method_LSTM.initialize(n=INPUT_DIM, m=1, l=SEQUENCE_LENGTH, h=HIDDEN_DIM, optimizer=optim, dp_rate=DP_RATE)
method_LSTM.load('trained_CAMEL_ogd_0.1_3900.pkl')
results_LSTM = []
pred_LSTM = []

def usgs_eval(method, site_idx, batch_size=None, dynamic=False, path=None):
	### We will do this with the NE metric
        optim = OGD(loss=batched_mse, learning_rate=LR)
        # optim = Adam(loss=batched_mse, learning_rate=LR, include_x_loss=False, hyperparameters=hyperparams)
        if dynamic:
                optim = DynamicEval(optim, prior_weight=0.0)
        eval_method = tigerforecast.method("Seq2ValLSTM")
        eval_method.initialize(n=INPUT_DIM, m=1, l=SEQUENCE_LENGTH, h=HIDDEN_DIM, optimizer=optim, dp_rate=0.0)
        if not path:
                eval_method.params = method.copy()
        else:
                eval_method.load(path)
        if dynamic:
                dynamic_optim.set_prior(eval_method.params)

        yhats, ys, ysf = [], [], []
        for data, targets in usgs_val.sequential_batches(site_idx=site_idx, batch_size=batch_size):
                y_pred = eval_method.predict(data[:,:,INDICES_TO_KEEP])
                targets_exp = np.expand_dims(targets[:,-1], axis=-1)
                if dynamic:
                        eval_method.update(targets_exp)
                yhats.append(y_pred[:,-1].copy())
                ys.append(targets[:,-1].copy())
                # ysf.append(targets[:,-1][0])
                #print("targets[:,-1] = " + str(targets[:,-1]))
                #print("targets[0,-1] = " + str(targets[0,-1]))
                #print("y_pred[:,-1] = " + str(y_pred[:,-1]))
                #print("y_pred[0,-1] = " + str(y_pred[0,-1]))
                # print("targets.shape")
                # print(targets.shape)
                # print("y_pred.shape")
                # print(y_pred.shape)
                # print("y_pred[0,-1] = " + str(type(y_pred[0,-1])))
                # print("y_pred[:,-1] = " + str(type(y_pred[:,-1])))
                # print("y_pred[:,-1][0] = " + str(type(y_pred[:,-1][0])))
                # print("targets[0,-1] = " + str(type(targets[0,-1])))
                # print("targets[:,-1] = " + str(type(targets[:,-1])))
                # print("targets[:,-1][0] = " + str(type(targets[:,-1][0])))
        # assert(len(yhats[0]) == 1024)
        return np.array(yhats), np.array(ys)

best_index = -1
best_loss = 100
best_nse = -1
best_index_no_outliers = -1
best_loss_no_outliers = 100
best_nse_no_outliers = -1
all_sites = usgs_train.get_all_sites()
sites_100 = rnd.sample(all_sites, 100)
for i, (data, targets) in enumerate( usgs_train.random_batches(batch_size=BATCH_SIZE, num_batches=TRAINING_STEPS) ):
	# print("predicting...")
	y_pred_LSTM = method_LSTM.predict(data[:,:,INDICES_TO_KEEP])
	# print("predicting complete")
	#pred_LSTM.append(y_pred_LSTM[0,-1,0])
	#print(y_pred_LSTM[0,:,0])
	#print(targets[0,:])
	if TRAIN_SEQ_TO_SEQ:
		targets_exp = np.expand_dims(targets, axis=-1)
	else:
		targets_exp = np.expand_dims(targets[:,-1], axis=-1)		
	loss = float(batched_mse(y_pred_LSTM, targets_exp))
	results_LSTM.append(loss)
	# print("updating...")
	method_LSTM.update(targets_exp)
	# print("updating complete")
	
	if i % 100 == 0:
                print("i = " + str(i))
	if i%300 == 0 and i > 0:
                method_LSTM.save('trained_CAMEL_ogd_'+ str(LR) + '_' + str(i) + '.pkl')
                print('Step %i: loss=%f' % (i,results_LSTM[-1]) )
                losses = []
                losses_no_outliers = []
                nses = []
                nses_no_outliers = []
                for site in sites_100:
                        yhats, ys = usgs_eval(method_LSTM, site, batch_size=1024, dynamic=False)
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
                avg_loss = np.array(losses).mean()
                avg_loss_no_outliers = np.array(losses_no_outliers).mean()
                avg_nse = np.array(nses).mean()
                avg_nse_no_outliers = np.array(nses_no_outliers).mean()
                print("avg_loss = " + str(avg_loss))
                print("avg_nse = " + str(avg_nse))
                print("avg_nse_no_outliers = " + str(avg_nse_no_outliers))
                if avg_nse > best_nse:
                        best_loss = avg_loss
                        best_index = i
                        best_nse= avg_nse
                if avg_nse_no_outliers > best_nse_no_outliers:
                        best_loss_no_outliers = avg_loss_no_outliers
                        best_index_no_outliers = i
                        best_nse_no_outliers = avg_nse_no_outliers
	if i == 6000:
		break

# best_index = 2200
# handle eval metrics
# best_index_no_outliers = 3900
# print("best_index = " + str(best_index))
print("best_index_no_outliers = " + str(best_index_no_outliers))
# print("best_loss_no_outliers = " + str(best_loss_no_outliers))
# print("best_nse_no_outliers = " + str(best_nse_no_outliers))
# print("best_loss = " + str(best_loss))
# print("best_nse = " + str(best_nse))
best_path = 'trained_CAMEL_ogd_' + str(LR) + "_" + str(best_index_no_outliers) + '.pkl'
all_sites = usgs_val.get_all_sites()
# sites_50 = rnd.sample(all_sites, 50)
eval_losses = []
nses = []
nses_no_outliers = []
cnt = 0
for site in all_sites:
        print("site = " + str(site))
        print(" =========== pkl eval ============ ")
        yhats, ys = usgs_eval(method_LSTM, site, batch_size=1024,dynamic=False, path=best_path)
        # yhatsn, ysn = usgs_eval(method_LSTM, site, batch_size=1, dynamic=False, path_best_path)
        # print("yhats[:10] = " + str(yhats[:10]))
        # print("yhats.flatten()[:10] = " + str(yhats.flatten()[:10]))
        # print("ys[:10] = " + str(ys[:10]))
        # print("ys.flatten()[:10] = " + str(ys.flatten()[:10]))
        # print("fgh[:10] = " + str(fgh[:10]))
        # print("jkl[:10] = " + str(jkl[:10]))
        # print("yhats.shape = " + str(yhats.shape))
        # print("ys.shape = " + str(ys.shape))
        loss = ((ys-yhats)**2).mean()
        print("loss = " + str(loss))
        eval_losses.append(loss)
        ys_mean = ys.mean()
        # print("ys_mean = " + str(ys_mean))
        # ysf = ys.flatten()
        # yhatsf = yhats.flatten()
        # ysf_mean = ysf.mean()
        # print("ys - ys_mean = " + str((ys-ys_mean)[:10]))
        # print("ys-yhats = " + str((ys-yhats)[:10]))
        # print("den = " + str(((ys - ys_mean)**2).sum()))
        # print("num = " + str(((ys - yhats)**2).sum()))
        nse = 1 - ((((ys - yhats)**2).sum())/(((ys - ys_mean)**2).sum()))
        # nsef = 1 - ((((ysf - yhatsf)**2).sum())/(((ysf - ysf_mean)**2).sum()))
        print("nse = " + str(nse))
        # print("nsef = " + str(nsef))
        if not np.isnan(nse):
                nses.append(nse)
        if not np.isnan(nse) and np.abs(nse) < 2:
                nses_no_outliers.append(nse)
        print("mean NSE so far: " + str(np.array(nses).mean()))
        print("mean NSE no outliers so far:" + str(np.array(nses_no_outliers).mean()))
        cnt += 1
        print("cnt = " + str(cnt))
        '''print(" =========== pkl eval ============= ")
        yhats, ys = usgs_eval(method_LSTM, site, dynamic=False, path=MODEL_PATH)
        loss = ((ys-yhats)**2).mean()
        print("loss = " + str(loss))
        ys_mean = ys.mean()
        nse = (1 - ((ys- yhats)**2).sum()/((ys - ys_mean)**2).sum())
        print("nse = " + str(nse))'''
        
# print('Eval: loss=%f' % ((ys-yhats)**2).mean() )
# ys_mean = ys.mean()
print('head of eval losses: ' + str(eval_losses[:10]))
print('mean NSE: %f' % np.array(nses).mean())
print('mean NSE no outliers: %f' % np.array(nses_no_outliers).mean())



	# if i == 1100:
	# 	optim.lr /= 10

print("Training Done")
# yhats, ys = usgs_eval(method_LSTM, 0)
# print(yhats.shape, ys.shape)

# plt.plot(yhats, 'b', label='predicted')
# plt.plot(ys, 'k', label='actual')
# plt.legend()
# plt.show()


# print(results_LSTM)

plt.subplot(121)
plt.plot(results_LSTM, label = 'LSTM')


plt.subplot(122)
plt.plot(pred_LSTM, label="prediction")

plt.legend()
plt.title("Seq to Seq LSTM Training on Flood problem")
plt.show(block=True)
