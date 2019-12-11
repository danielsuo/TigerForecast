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
import numpy as onp

_METRICS_CLIPPED_RMSE_RETURN_PERIOD = 0.005  # Roughly 50% of the data.
_METRICS_RETURN_PERIODS = (0.05, 0.1, 1.)


TRAINING_STEPS = 100
BATCH_SIZE = 1024
SEQUENCE_LENGTH = 61
HIDDEN_DIM = 100
EMBEDDING_DIM = 10
DATA_PATH = '../data/usgs_flood/usgs_{}.csv'
DATA_METRICS_PATH = '../data/usgs_flood/usgs_{}.csv'

# optim = OGD(loss=batched_mse, learning_rate=0.1)
hyperparams = {'reg':0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'eps': 1e-8, 'max_norm':True}
optim = Adam(loss=batched_mse, learning_rate=1.0, hyperparameters=hyperparams)

usgs_train = USGSDataLoader(path=DATA_PATH.format('train_mini'), 
							metrics_path = DATA_PATH.format('train_threshold_data'))
usgs_val = USGSDataLoader(path=DATA_PATH.format('val_mini'), 
						  metrics_path=DATA_PATH.format('val_threshold_data'), 
						  site_idx=usgs_train.site_idx, 
						  normalize_source=usgs_train)

method_LSTM = tigerforecast.method("FloodLSTM")
method_LSTM.initialize(n=8, m=1, l=61, h=HIDDEN_DIM, e_dim=EMBEDDING_DIM, num_sites=len(usgs_train.site_keys), optimizer=optim, dp_rate=0.1)

results_LSTM = []
pred_LSTM = []

def calculate_r2_per_site(y_true, y_pred):
	"""Calculates the R2 per site.
	Args:
	y_true: The true labels of a single site.
	y_pred: The predictions (must match y_true.shape).
	weights: NOT SUPPORTED.
	Returns:
	R2 for this site's data.
	"""
	y_true = np.array(y_true, dtype=np.float32)
	y_pred = np.array(y_pred, dtype=np.float32)
	ss_res = ((y_true - y_pred) ** 2).sum()
	ss_tot = ((y_true - y_true.mean()) ** 2).sum()
	# Arbitrarily set to 0.0 to avoid -inf scores. Also the sklearn behavior.
	if ss_tot == 0:
		return 0.
	else:
		return 1 - ss_res / ss_tot

def median_r2_per_site(targets, preds, site_ids):
	site_id_to_target_pred_series = {}
	for i in range(len(site_ids)):
		if site_ids[i] not in site_id_to_target_pred_series:
			site_id_to_target_pred_series[site_ids[i]] = ([], [])
		site_id_to_target_pred_series[site_ids[i]][0].append(targets[i])
		site_id_to_target_pred_series[site_ids[i]][1].append(preds[i])
	
	all_r2s = []
	for site_id, (targets, preds) in site_id_to_target_pred_series.items():
		all_r2s.append(calculate_r2_per_site(targets, preds))
	return np.median(all_r2s)

def filtered_mape_metric(targets, preds, thresholds):
	"""Calculates filtered MAPE.

	MAPE (mean absolute percentage error) measures the diff between labels and
	predictions, normalized by the label. (The error in % of the true label).
	Filtered-MAPE will only be reported for examples with label greater than a
	given threshold. This metric should give a sense of how close are we to the
	actual label in the "interesting" zone.

	Args:
	labels: A float 1D `Tensor`.
	predictions: A float 1D `Tensor` whose shape matches `labels`.
	thresholds: A float 1D `Tenor` whose shape matches `labels'.
	Returns:
	A tensor of filtered_mape
	"""
	mask = np.greater_equal(targets, thresholds)
	filtered_targets = onp.extract(mask, targets)
	filtered_preds = onp.extract(mask, preds)
	filtered_targets_preds = list(zip(filtered_targets, filtered_preds))
	mape = lambda t_p : abs((t_p[0]-t_p[1])/t_p[0])
	maped = jax.vmap(mape)(np.array(filtered_targets_preds))
	return np.mean(maped)
	# summer = lambda cnt_sum, x: ((cnt_sum[0]+1, cnt_sum[1]+x), 0) if x != 0 else (cnt_sum, 0)
	# cnt_sum, _ = jax.lax.scan(summer, 0, maped)
	# return cnt_sum[1]/cnt_sum[0]

def compute_precision_recall(targets_binary, preds_binary):
	targets_preds_binary = np.array(list(zip(targets_binary, preds_binary)))
	TP_finder = lambda t_b : t_b[0] * t_b[1]
	FP_finder = lambda t_b : (1 - t_b[0]) * t_b[1]
	FN_finder = lambda t_b: t_b[0] * (1 - t_b[1]) 
	TP = np.sum(jax.vmap(TP_finder)(targets_preds_binary))
	FP = np.sum(jax.vmap(FP_finder)(targets_preds_binary))
	FN = np.sum(jax.vmap(FN_finder)(targets_preds_binary))
	precision = TP/(TP+FP) if TP+FP > 0 else 0
	recall = TP/(TP+FN) if TP+FN > 0 else 0
	return precision, recall

def binary_classification_metric(targets, preds, thresholds):
	"""Calculates precision/recall of a thresholded binary classification problem.

	Converts a regression problem to a binary classification problem using
	thresholds per example (we noramlly precalculate several frequency values for
	each site and attach this vector to every example of this site). Values that
	are greater or equal to the threshold will be considered True, others - False.

	Args:
	labels: A float 1D `Tensor`.
	predictions: A float 1D `Tensor` whose shape matches `labels`.
	thresholds: A float 1D `Tenor` whose shape matches `labels'.
	  Will be used as the threshold to convert labels/predictions into a binary
	  classification problem.
	Returns:
	A tuple of (precision, recall).
	precision: Scalar float `Tensor` with the precision value.
	recall: Scalar float `Tensor` with the recall value.
	"""
	targets_binary = np.greater_equal(targets, thresholds).astype(int)
	preds_binary = np.greater(preds, thresholds).astype(int)
	return compute_precision_recall(targets_binary, preds_binary)

def trigger_binary_classification_metric(targets, yesterdays_targets, preds, thresholds):
	"""Calculates metrics for the "is_trigger" binary classification.

	An example is defined as a "trigger" if its label is above a certain
	threshold and yesterday's label is below. We calculate precision/recall on
	whether we successfully predicted the trigger.

	A prediction is defined as a "trigger" if its value is above a certain
	threshold and yesterday's **label** is below (we run the model on a single
	day, therefore there is no "yesterday's prediction").

	Calculated metrics are: Precision, Recall and F1 score.

	Args:
	labels: A float 1D `Tensor`.
	yesterdays_labels: A float 1D `Tensor` whose shape matches `labels`.
	predictions: A float 1D `Tensor` whose shape matches `labels`.
	thresholds: A float 1D `Tenor` whose shape matches `labels'. Will be used as
	  the threshold to decide on a trigerring event.
	Returns:
	A tuple of (precision, recall, f1)
	"""
	targets_is_trigger = np.logical_and(
		np.greater_equal(targets, thresholds),
		np.less(yesterdays_targets, thresholds)).astype(int)
	preds_is_trigger = np.logical_and(
		np.greater_equal(preds, thresholds),
		np.less(yesterdays_targets, thresholds)).astype(int)

	precision, recall = compute_precision_recall(targets_is_trigger, preds_is_trigger)

	f1 = 2 * precision * recall / (precision + recall) if precision+recall > 0 else 0

	return precision, recall, f1


def metrics_per_return_period(ys, y_hats, yesterdays_ys, thresholds, dataset):
	mape = filtered_mape_metric(ys, y_hats, thresholds)
	print('    %s: mape=%f' % (dataset,mape))

	precision, recall = binary_classification_metric(ys, y_hats, thresholds)
	print('    %s: precision=%f' % (dataset,precision))
	print('    %s: recall=%f' % (dataset, recall))

	trigger_precision, trigger_recall, f1 = trigger_binary_classification_metric(ys, yesterdays_ys, y_hats, thresholds)
	print('    %s: f1=%f' % (dataset, f1))
	print('    %s: trigger precision=%f' % (dataset, trigger_precision))
	print('    %s: trigger recall=%f' % (dataset, trigger_recall))

	return mape, precision, recall, trigger_precision, trigger_recall, f1


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
	site_ids = []
	yesterdays_ys = []
	for data, targets in usgs_val.sequential_batches(site_idx=1, batch_size=1):
		y_pred = eval_method.predict(data)

		targets_exp = np.expand_dims( np.expand_dims(targets[:,-1], axis=-1), axis=-1 )
		if dynamic:
			eval_method.update(targets_exp)

		yhats.append(y_pred[0,-1,0])
		ys.append(targets[0,-1])
		site_ids.append(data[0,0,0])
		yesterdays_ys.append(targets[0,-2])

	return np.array(yhats), np.array(ys), site_ids, np.array(yesterdays_ys)

train_median_r2_per_site = []
train_mape = {}
train_precision = {}
train_recall = {}
train_trigger_precision = {}
train_trigger_recall = {}
train_f1 = {}

eval_median_r2_per_site = []
eval_mape = {}
eval_precision = {}
eval_recall = {}
eval_trigger_precision = {}
eval_trigger_recall = {}
eval_f1 = {}
for return_period in _METRICS_RETURN_PERIODS:
	eval_mape[return_period] = []
	eval_precision[return_period] = []
	eval_recall[return_period] = []
	eval_trigger_precision[return_period] = []
	eval_trigger_recall[return_period] = []
	eval_f1[return_period] = []

	train_mape[return_period] = []
	train_precision[return_period] = []
	train_recall[return_period] = []
	train_trigger_precision[return_period] = []
	train_trigger_recall[return_period] = []
	train_f1[return_period] = []

for i, (data, targets) in enumerate( usgs_train.random_batches(batch_size=BATCH_SIZE, num_batches=TRAINING_STEPS) ):
	y_pred_LSTM = method_LSTM.predict(data)
	pred_LSTM.append(y_pred_LSTM[0,-1,0])
	#print(y_pred_LSTM[0,:,0])
	#print(targets[0,:])
	targets_exp = np.expand_dims(targets, axis=-1)
	loss = float(batched_mse(jax.device_put(targets_exp), y_pred_LSTM))
	results_LSTM.append(loss)
	method_LSTM.update(targets_exp)

	y_hats = y_pred_LSTM[:,-1,0]
	ys = targets[:,-1]
	yesterdays_ys = targets[:,-2]

	if i%100 == 0:
		print("========================================")
		print('Step %i: loss=%f' % (i,results_LSTM[-1]) )

		# handle training metrics

		site_ids = data[:,0,0]
		med_r2_per_site = median_r2_per_site(ys, y_hats, site_ids)
		print('Train: median_r2_per_site=%f' % med_r2_per_site)
		train_median_r2_per_site.append(med_r2_per_site)

		for return_period in _METRICS_RETURN_PERIODS:
			print("return period=%f" % return_period)
			thresholds = []
			for site_id in site_ids:
				data_metrics = usgs_train.get_data_metrics(site_idx=site_id)
				return_period_value = data_metrics[1][list(data_metrics[2]).index(return_period)]
				thresholds.append(return_period_value)
			thresholds = np.array(thresholds)

			metrics = metrics_per_return_period(ys, y_hats, yesterdays_ys, thresholds, 'train')
			mape, precision, recall, trigger_precision, trigger_recall, f1 = metrics
			train_mape[return_period].append(mape)
			train_precision[return_period].append(precision)
			train_recall[return_period].append(recall)
			train_trigger_precision[return_period].append(trigger_precision)
			train_trigger_recall[return_period].append(trigger_recall)
			train_f1[return_period].append(f1)

		# handle eval metrics
		
		yhats, ys, site_ids, yesterdays_ys = usgs_eval(method_LSTM, 0, dynamic=False)
		print('Eval: loss=%f' % ((ys-yhats)**2).mean() )

		med_r2_per_site = median_r2_per_site(ys, yhats, site_ids)
		print('Eval: median_r2_per_site=%f' % med_r2_per_site)
		eval_median_r2_per_site.append(med_r2_per_site)

		data_metrics = usgs_val.get_data_metrics(site_idx=1)
		# print("data_metrics:")
		# print(data_metrics)
		for return_period in _METRICS_RETURN_PERIODS:
			print("return period=%f" % return_period)
			return_period_value = data_metrics[1][list(data_metrics[2]).index(return_period)]
			thresholds = np.repeat(return_period_value, len(site_ids))
			metrics = metrics_per_return_period(ys, yhats, yesterdays_ys, thresholds, 'eval')
			mape, precision, recall, trigger_precision, trigger_recall, f1 = metrics
			eval_mape[return_period].append(mape)
			eval_precision[return_period].append(precision)
			eval_recall[return_period].append(recall)
			eval_trigger_precision[return_period].append(trigger_precision)
			eval_trigger_recall[return_period].append(trigger_recall)
			eval_f1[return_period].append(f1)

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

for return_period in _METRICS_RETURN_PERIODS:
	p = str(return_period)
	plt.subplot(231)
	plt.plot(train_mape[return_period], label="train_mape_" + p)
	plt.legend()

	plt.subplot(232)
	plt.plot(train_precision[return_period], label="train_precision_" + p)
	plt.legend()

	plt.subplot(233)
	plt.plot(train_recall[return_period], label="train_recall" + p)
	plt.legend()

	plt.subplot(234)
	plt.plot(train_f1[return_period], label="train_f1" + p)
	plt.legend()

	plt.subplot(235)
	plt.plot(train_trigger_precision[return_period], label="train_trigger_precision_" + p)
	plt.legend()

	plt.subplot(236)
	plt.plot(train_trigger_recall[return_period], label="train_trigger_recall_" + p)
	plt.legend()
	
	plt.show(block=True)

for return_period in _METRICS_RETURN_PERIODS:
	p = str(return_period)
	plt.subplot(231)
	plt.plot(eval_mape[return_period], label="eval_mape_" + p)
	plt.legend()

	plt.subplot(232)
	plt.plot(eval_precision[return_period], label="eval_precision_" + p)
	plt.legend()

	plt.subplot(233)
	plt.plot(eval_recall[return_period], label="eval_recall" + p)
	plt.legend()

	plt.subplot(234)
	plt.plot(eval_f1[return_period], label="eval_f1" + p)
	plt.legend()

	plt.subplot(235)
	plt.plot(eval_trigger_precision[return_period], label="eval_trigger_precision_" + p)
	plt.legend()

	plt.subplot(236)
	plt.plot(eval_trigger_recall[return_period], label="eval_trigger_recall_" + p)
	plt.legend()

	plt.show(block=True)

plt.close()


