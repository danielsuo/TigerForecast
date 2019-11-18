import tigerforecast
import jax
import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
import pickle
from tigerforecast.utils import generate_key
from usgs_data_loader import *

TRAINING_STEPS = 100
BATCH_SIZE= 100
SEQUENCE_LENGTH = 61
HIDDEN_DIM = 100
EMBEDDING_DIM = 10
DATA_PATH = 'usgs_{}_mini.csv'

loss = lambda pred, true: np.mean(np.sum((pred - true)**2, axis=(1,2)))

usgs_train = USGSDataLoader(DATA_PATH.format('train'))
usgs_val = USGSDataLoader(DATA_PATH.format('val'), site_idx=usgs_train.site_idx)

method_LSTM = tigerforecast.method("FloodLSTM")
method_LSTM.initialize(n=9, m=1, l = 61, h = HIDDEN_DIM, e_dim = EMBEDDING_DIM, num_sites = len(usgs_train.site_keys), optimizer = None)

results_LSTM = []
pred_LSTM = []
for data, targets in usgs_train.random_batches(batch_size=BATCH_SIZE, num_batches=TRAINING_STEPS):
	y_pred_LSTM = method_LSTM.predict(data)
	pred_LSTM.append(y_pred_LSTM[0,-1,0])
	#print(y_pred_LSTM[0,:,0])
	#print(targets[0,:])
	targets_exp = np.expand_dims(targets, axis=-1)
	results_LSTM.append(loss(targets_exp, y_pred_LSTM))
	method_LSTM.update(targets_exp)

print("Training Done")

# def usgs_eval(method, site_idx):
# 	yhats, ys = [], []
# 	for data, targets in usgs_train.sequential_batches(site_idx=1, batch_size=1):
# 		y_pred_LSTM = method.predict(data)
# 		yhats.append(y_pred_LSTM[0,-1,0])
# 		ys.append(targets[0,-1])
# 	return np.array(yhats), np.array(ys)

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


