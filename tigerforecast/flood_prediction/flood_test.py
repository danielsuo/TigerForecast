import tigerforecast
import jax
import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
import pickle
from tigerforecast.utils import generate_key
from tigerforecast.batch.usgs import *

from tigerforecast.utils.optimizers import *
from tigerforecast.utils.optimizers.losses import *
from tigerforecast.utils.download_tools import *

TRAINING_STEPS = 2000
BATCH_SIZE = 1024
SEQUENCE_LENGTH = 61
HIDDEN_DIM = 100
EMBEDDING_DIM = 10
DATA_PATH = '/home/cyrilzhang/data/usgs_{}.csv'
tigerforecast_dir = get_tigerforecast_dir()
train_mini_path = os.path.join(tigerforecast_dir, 'data/usgs_flood/usgs_train_mini.csv')
val_mini_path = os.path.join(tigerforecast_dir, 'data/usgs_flood/usgs_val_mini.csv')
optim = Adam(loss=batched_mse, learning_rate=1.0)

# usgs_train = USGSDataLoader(DATA_PATH.format('train_mini'))
usgs_train = USGSDataLoader(train_mini_path)
usgs_val = USGSDataLoader(val_mini_path, site_idx=usgs_train.site_idx, normalize_source=usgs_train)
# usgs_val = USGSDataLoader(DATA_PATH.format('val_mini'), site_idx=usgs_train.site_idx, normalize_source=usgs_train)

method_LSTM = tigerforecast.method("FloodLSTM")
method_LSTM.initialize(n=8, m=1, l = 61, h = HIDDEN_DIM, e_dim = EMBEDDING_DIM, num_sites = len(usgs_train.site_keys), optimizer=optim, dp_rate=0.1)

results_LSTM = []
pred_LSTM = []

def usgs_eval(method, site_idx):
	yhats, ys = [], []
	for data, targets in usgs_val.sequential_batches(site_idx=1, batch_size=1):
		y_pred_LSTM = method.predict(data)
		yhats.append(y_pred_LSTM[0,-1,0])
		ys.append(targets[0,-1])
	return np.array(yhats), np.array(ys)


for i, (data, targets) in enumerate( usgs_train.random_batches(batch_size=BATCH_SIZE, num_batches=TRAINING_STEPS) ):
	y_pred_LSTM = method_LSTM.predict(data)
	pred_LSTM.append(y_pred_LSTM[0,-1,0])
	#print(y_pred_LSTM[0,:,0])
	#print(targets[0,:])
	targets_exp = np.expand_dims(targets, axis=-1)
	loss = float(batched_mse(jax.device_put(targets_exp), y_pred_LSTM))
	results_LSTM.append(loss)
	method_LSTM.update(targets_exp)

	if i%100 == 0:
		print('Step %i: loss=%f' % (i,results_LSTM[-1]) )
	if i%400 == 0 or i+1 == TRAINING_STEPS:
		yhats, ys = usgs_eval(method_LSTM, 0)
		print('Eval: loss=%f' % ((ys-yhats)**2).mean() )

