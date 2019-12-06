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

# optim = OGD(loss=batched_mse, learning_rate=0.1)
hyperparams = {'reg':0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'eps': 1e-8, 'max_norm':True}
optim = Adam(loss=batched_mse, learning_rate=1.0, hyperparameters=hyperparams)

usgs_train = USGSDataLoader(DATA_PATH.format('train_mini'))
usgs_val = USGSDataLoader(DATA_PATH.format('val_mini'), site_idx=usgs_train.site_idx, normalize_source=usgs_train)

method_LSTM = tigerforecast.method("FloodLSTM")
method_LSTM.initialize(n=8, m=1, l=61, h=HIDDEN_DIM, e_dim=EMBEDDING_DIM, num_sites=len(usgs_train.site_keys), optimizer=optim, dp_rate=0.1)

results_LSTM = []
pred_LSTM = []

def usgs_eval(method, site_idx, dynamic=False):
	optim = OGD(loss=batched_mse, learning_rate=0.1)
	if dynamic:
		optim = DynamicEval(optim, prior_weight=0.0, exclude={'W_embed'})

	eval_method = tigerforecast.method("FloodLSTM")
	eval_method.initialize(n=8, m=1, l=61, h=HIDDEN_DIM, e_dim=EMBEDDING_DIM, num_sites=len(usgs_train.site_keys), optimizer=optim, dp_rate=0.0)
	eval_method.params = method.copy()

	if dynamic:
		dynamic_optim.set_prior(eval_method.params)

	yhats, ys = [], []
	for data, targets in usgs_val.sequential_batches(site_idx=1, batch_size=1):
		y_pred = eval_method.predict(data)

		targets_exp = np.expand_dims( np.expand_dims(targets[:,-1], axis=-1), axis=-1 )
		if dynamic:
			eval_method.update(targets_exp)

		yhats.append(y_pred[0,-1,0])
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
		yhats, ys = usgs_eval(method_LSTM, 0, dynamic=False)
		print('Eval: loss=%f' % ((ys-yhats)**2).mean() )

	if i == 1100:
		optim.lr /= 10

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
plt.close()


