# loss functions

import jax.numpy as np 

def mse(y_pred, y_true):
    ''' Description: mean-square-error loss
        Args:
            y_pred : value predicted by method
            y_true : ground truth value
            eps: some scalar
    '''
    return np.sum((y_pred - y_true)**2)
    
def cross_entropy(y_pred, y_true, eps=1e-9):
    ''' Description: cross entropy loss, y_pred is equivalent to logits and y_true to labels
        Args:
            y_pred : value predicted by method
            y_true : ground truth value
            eps: some scalar
    '''
    return - np.sum(y_true * np.log(y_pred + eps))

def batched_mse(y_pred, y_true):
    ''' Description: mean-square-error loss on a batch
        Args:
            y_pred : value predicted by method
            y_true : ground truth value
    '''
    return np.mean(np.sum((y_pred - y_true)**2, axis=tuple(range(1, y_true.ndim))))

def batched_mse_flood_adjusted(y_pred, x, y_true):
    ''' Description: mean-square-error loss on a batch adjusting on site std for flood
        Args:
            y_pred : value predicted by method
            x : input
            y_true : ground truth value
    '''
    example_wise_loss = np.mean((y_pred - y_true)**2, axis=tuple(range(1, y_true.ndim)))
    rescale_vector = 1/(x[0][:,-1,-1] + 0.1)**2
    return np.mean(rescale_vector*example_wise_loss)

def batched_mse_flood_adjusted_withoutDO(y_pred, x, y_true):
    ''' Description: mean-square-error loss on a batch adjusting on site std for flood
        Args:
            y_pred : value predicted by method
            x : input
            y_true : ground truth value
    '''
    example_wise_loss = np.mean((y_pred - y_true)**2, axis=tuple(range(1, y_true.ndim)))
    # print("x.shape = " + str(x.shape))
    rescale_vector = 1/(x[:,-1,-1] + 0.1)**2
    return np.mean(rescale_vector*example_wise_loss)


