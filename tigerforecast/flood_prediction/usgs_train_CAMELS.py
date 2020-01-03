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


TRAINING_STEPS = 100000
BATCH_SIZE = 1024
SEQUENCE_LENGTH = 270
HIDDEN_DIM = 256
EMBEDDING_DIM = 10
DP_RATE = 0.4
INDICES_TO_KEEP = [15, 16, 17, 18, 19, 20, 21, 22, 26, 27, 28, 29, 30, 31, 32, 34, 35, 37, 38, 39, 40, 41, 48, 49, 50, 56, 58, 60, 61, 62, 63, 64, 65]
INPUT_DIM = len(INDICES_TO_KEEP)
DATA_PATH = '../data/usgs_flood/usgs_{}.csv'
TRAIN_SEQ_TO_SEQ = False

# optim = OGD(loss=batched_mse, learning_rate=0.1)
hyperparams = {'reg':0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'eps': 1e-8, 'max_norm':True}
optim = Adam(loss=batched_mse, learning_rate=1e-3, include_x_loss=False, hyperparameters=hyperparams)

usgs_train = USGSDataLoader(mode='train')
usgs_val = USGSDataLoader(mode='val', gaugeID_to_idx=usgs_train.gaugeID_to_idx)

if TRAIN_SEQ_TO_SEQ:
	method_LSTM = tigerforecast.method("Seq2SeqLSTM")
else:
	method_LSTM = tigerforecast.method("Seq2ValLSTM")
method_LSTM.initialize(n=INPUT_DIM, m=1, l=SEQUENCE_LENGTH, h=HIDDEN_DIM, optimizer=optim, dp_rate=DP_RATE)

results_LSTM = []
pred_LSTM = []

def usgs_eval(method, site_idx, dynamic=False):
	### We will do this with the NE metric
	optim = OGD(loss=batched_mse, learning_rate=0.1)
	if dynamic:
		optim = DynamicEval(optim, prior_weight=0.0)

	eval_method = tigerforecast.method("Seq2ValLSTM")
	eval_method.initialize(n=INPUT_DIM, m=1, l=SEQUENCE_LENGTH, h=HIDDEN_DIM, optimizer=optim, dp_rate=0.0)
	eval_method.params = method.copy()

	if dynamic:
		dynamic_optim.set_prior(eval_method.params)

	yhats, ys = [], []
	for data, targets in usgs_val.sequential_batches(site_idx=1, batch_size=1):
		y_pred = eval_method.predict(data[:,:,INDICES_TO_KEEP])
		targets_exp = np.expand_dims(targets[:,-1], axis=-1)
		if dynamic:
			eval_method.update(targets_exp)

		yhats.append(y_pred[0,-1])
		ys.append(targets[0,-1])

	return np.array(yhats), np.array(ys)

for i, (data, targets) in enumerate( usgs_train.random_batches(batch_size=BATCH_SIZE, num_batches=TRAINING_STEPS) ):
	y_pred_LSTM = method_LSTM.predict(data[:,:,INDICES_TO_KEEP])
	#pred_LSTM.append(y_pred_LSTM[0,-1,0])
	#print(y_pred_LSTM[0,:,0])
	#print(targets[0,:])
	if TRAIN_SEQ_TO_SEQ:
		targets_exp = np.expand_dims(targets, axis=-1)
	else:
		targets_exp = np.expand_dims(targets[:,-1], axis=-1)		
	loss = float(batched_mse(y_pred_LSTM, targets_exp))
	results_LSTM.append(loss)
	method_LSTM.update(targets_exp)
	
	if i%100 == 0:
		print('Step %i: loss=%f' % (i,results_LSTM[-1]) )
	if i%1000 == 0:
		# handle eval metrics	
		yhats, ys = usgs_eval(method_LSTM, 0, dynamic=False)
		print('Eval: loss=%f' % ((ys-yhats)**2).mean() )
		ys_mean = ys.mean()
		print('NSE: %f' % (1 - ((ys - yhats)**2).sum()/((ys - ys_mean)**2).sum()))



	# if i == 1100:
	# 	optim.lr /= 10

print("Training Done")
# yhats, ys = usgs_eval(method_LSTM, 0)
# print(yhats.shape, ys.shape)

# plt.plot(yhats, 'b', label='predicted')
# plt.plot(ys, 'k', label='actual')
# plt.legend()
# plt.show()


print(results_LSTM)

plt.subplot(121)
plt.plot(results_LSTM, label = 'LSTM')


plt.subplot(122)
plt.plot(pred_LSTM, label="prediction")

plt.legend()
plt.title("Seq to Seq LSTM Training on Flood problem")
plt.show(block=True)
