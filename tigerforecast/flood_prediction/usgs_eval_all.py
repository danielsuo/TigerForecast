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

SEQUENCE_LENGTH = 61
HIDDEN_DIM = 100
EMBEDDING_DIM = 10
DATA_PATH = '../data/usgs_flood/usgs_{}.csv'
MODEL_PATH = 'full_99000.npy'

usgs_train = USGSDataLoader(DATA_PATH.format('train'))
usgs_val = USGSDataLoader(DATA_PATH.format('val'), site_idx=usgs_train.site_idx, normalize_source=None)

def usgs_eval(path, site_idx, dynamic=False):
	optim = OGD(loss=batched_mse, learning_rate=0.1)
	if dynamic:
		optim = DynamicEval(optim, prior_weight=0.0, exclude={'W_embed'})

	eval_method = tigerforecast.method("FloodLSTM")
	eval_method.initialize(n=8, m=1, l=61, h=HIDDEN_DIM, e_dim=EMBEDDING_DIM, num_sites=len(usgs_train.site_keys), optimizer=optim, dp_rate=0.0)
	eval_method.load(path)

	if dynamic:
		dynamic_optim.set_prior(eval_method.params)

	yhats, ys, losses = [], [], []
	for data, targets in usgs_val.sequential_batches(site_idx=1, batch_size=1):
		y_pred = eval_method.predict(data)

		targets_exp = np.expand_dims( np.expand_dims(targets[:,-1], axis=-1), axis=-1 )
		if dynamic:
			eval_method.update(targets_exp)

		loss = float(batched_mse_flood_adjusted(y_pred, (data,), targets_exp) )

		yhats.append(y_pred[0,-1,0])
		ys.append(targets[0,-1])
		losses.append(loss)

	return np.array(yhats), np.array(ys), np.array(losses)

for idx, key in usgs_val.site_idx.items():
	yhats, ys, losses = usgs_eval(MODEL_PATH, idx, dynamic=False)
	print('Eval %i: mse=%f, seq_loss=%f' % (key, ((ys-yhats)**2).mean(), losses.mean()) )





