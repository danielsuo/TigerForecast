# test the seq2seq LSTM method class on ARMA

import tigerforecast
import jax
import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
import pickle
from tigerforecast.utils import generate_key



def test_seq2seq_lstm_arma(steps=100, show_plot=True):
    T = steps 
    p, q = 3, 0
    n, m = 1, 1
    batch_size = 32
    problem = tigerforecast.problem("ARMA-v0")
    _ = problem.initialize(p, q)


    method_LSTM = tigerforecast.method("Seq2seqLSTMStateless")
    method_LSTM.initialize(n, m, l = p)

    loss = lambda pred, true: np.mean(np.sum((pred - true)**2, axis=(1,2)))
 
    def get_batch():
        x = np.zeros((batch_size, p, m))
        for i in range(batch_size):
            for j in range(p):
                x = jax.ops.index_update(x, jax.ops.index[i,j,:], problem.step())
        return x                

    cur_x = get_batch()

    results_LSTM = []

    for i in range(T):
        y_pred_LSTM = method_LSTM.predict(cur_x)
        results_LSTM.append(loss(cur_x, y_pred_LSTM))
        print(y_pred_LSTM[0,-1,0])
        method_LSTM.update(cur_x)
        cur_x = get_batch()

    # print("Saving things")
    # method_LSTM.save("test.p")
    # new_method_LSTM = tigerforecast.method("Seq2seqLSTMStateless")
    # new_method_LSTM.initialize_with_ckpt(n, m, l = p, filename="test.p")

    # for i in range(T):
    #     y_pred_LSTM = new_method_LSTM.predict(cur_x)
    #     results_LSTM.append(loss(cur_x, y_pred_LSTM))
    #     new_method_LSTM.update(cur_x)
    #     cur_x = get_batch()

    if show_plot:
        plt.plot(results_LSTM, label = 'LSTM')
        plt.legend()
        plt.title("Seq to Seq LSTM on ARMA problem")
        plt.show(block=True)
        plt.pause(3)
        plt.close()
    print("test_seq2seq_lstm_arma passed")
    return

if __name__=="__main__":
    test_seq2seq_lstm_arma()