import tigerforecast
import jax
import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
import pickle
from tigerforecast.utils import generate_key
from usgs_data_loader import *

TRAINING_STEPS = 30000
BATCH_SIZE = 1024
SEQUENCE_LENGTH = 61
HIDDEN_DIM = 100
EMBEDDING_DIM = 10
DATA_PATH = 'usgs_{}_mini.csv'
FILENAME = 'full_400.pkl'

TEST_BOOSTING = False
TEST_ONLINE = True

loss = lambda pred, true: np.mean(np.sum((pred - true)**2, axis=(1,2)))
last_loss = lambda pred, true: np.mean( (pred[:,-1,:]-true[:,-1,:])**2 )

usgs_train = USGSDataLoader(DATA_PATH.format('train'))
usgs_val = USGSDataLoader(DATA_PATH.format('val'), site_idx=usgs_train.site_idx, normalize_source=usgs_train)

method_LSTM = tigerforecast.method("FloodLSTM")
method_LSTM.initialize(n=9, m=1, l = 61, h = HIDDEN_DIM, e_dim = EMBEDDING_DIM, num_sites = len(usgs_train.site_keys), optimizer = None, filename = FILENAME)
if TEST_ONLINE:
	online_LSTM = tigerforecast.method("FloodLSTM")
	online_LSTM.initialize(n=9, m=1, l = 61, h = HIDDEN_DIM, e_dim = EMBEDDING_DIM, num_sites = len(usgs_train.site_keys), optimizer = None, filename = FILENAME)
if TEST_BOOSTING:
	params_dict = {'l':61, 'h':HIDDEN_DIM, 'e_dim':EMBEDDING_DIM, 'num_sites':len(usgs_train.site_keys), 'optimizer':None, 'filename':FILENAME}
	boosted_LSTM = tigerforecast.method("SimpleBoost")
	boosted_LSTM.initialize("FloodLSTM",params_dict,n=9,m=1)
results_LSTM = []
pred_LSTM = []
online_pred_LSTM = []
boosted_pred_LSTM = []
def usgs_eval(method, site_idx,online=False):
	yhats, ys = [], []
	for data, targets in usgs_val.sequential_batches(site_idx=1, batch_size=1):
		y_pred_LSTM = method.predict(data)
		yhats.append(y_pred_LSTM[0,-1,0])
		ys.append(targets[0,-1])
		if online:
			method.update(np.expand_dims(targets, axis=-1))
	return np.array(yhats), np.array(ys)


yhats, ys = usgs_eval(method_LSTM, 0)
if TEST_ONLINE:
	oyhats, oys = usgs_eval(online_LSTM, 0, True)
if TEST_BOOSTING:
	print("Doing Boosting Eval")
	byhats, bys = usgs_eval(boosted_LSTM, 0, True)

residual = np.abs(yhats - ys)
print(np.mean(residual))
np.save('residual.npy', residual)
np.save('ys.npy', ys)

plt.semilogy(residual, 'b', label="Residual")
if TEST_ONLINE:
	oresidual = np.abs(oyhats -oys)
	print(np.mean(oresidual))
	print(np.mean(oresidual[ len(oresidual)//2: ]))
	np.save('online_residual.npy', oresidual)
	plt.semilogy(oresidual, 'k', label="Online Residual")

if TEST_BOOSTING:
	bresidual = np.abs(byhats -bys)
	print(np.mean(bresidual))
	plt.semilogy(bresidual, 'r', label="Boosted Online Residual")

plt.legend()
plt.show()



# plt.subplot(121)
# plt.plot(results_LSTM, label = 'LSTM')


# plt.subplot(122)
# plt.plot(pred_LSTM, label="prediction")

# plt.legend()
# plt.title("Seq to Seq LSTM Training on Flood problem")
# plt.show(block=True)
# plt.close()


