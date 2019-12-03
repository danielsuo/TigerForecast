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

TRAINING_STEPS = 30000
BATCH_SIZE = 1024
SEQUENCE_LENGTH = 61
HIDDEN_DIM = 100
EMBEDDING_DIM = 10
DATA_PATH = '../data/usgs_flood/usgs_{}.csv'

optim = OGD(loss=batched_mse, learning_rate=0.1)

usgs_train = USGSDataLoader(DATA_PATH.format('train_mini'))
usgs_val = USGSDataLoader(DATA_PATH.format('val_mini'), site_idx=usgs_train.site_idx, normalize_source=usgs_train)

method = tigerforecast.method("FloodAR")
method.initialize(n=8, m=1, l=61, num_sites=len(usgs_train.site_keys), optimizer=optim)

results = []
preds = []

def usgs_eval(method, site_idx):
	yhats, ys = [], []
	for data, targets in usgs_val.sequential_batches(site_idx=1, batch_size=1):
		y_pred = method.predict(data)
		yhats.append(y_pred[0,-1,0])
		ys.append(targets[0,-1])
	return np.array(yhats), np.array(ys)


for i, (data, targets) in enumerate( usgs_train.random_batches(batch_size=BATCH_SIZE, num_batches=TRAINING_STEPS) ):
	y_pred = method.predict(data)
	preds.append(y_pred[0,-1,0])
	targets_exp = np.expand_dims( np.expand_dims(targets[:,-1], axis=-1), axis=-1 )
	loss = float(batched_mse(jax.device_put(targets_exp), y_pred))
	results.append(loss)
	method.update(targets_exp)

	if i%100 == 0:
		print('Step %i: loss=%f' % (i,results[-1]) )
# if i%1000 == 0:
		yhats, ys = usgs_eval(method, 0)
		print('Eval: loss=%f' % ((ys-yhats)**2).mean() )
#		method.save('full_%i.pkl' % i)

