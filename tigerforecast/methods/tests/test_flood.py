# test the LSTM method class

import tigerforecast
import numpy as onp
import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
from tigerforecast.utils import generate_key
from tigerforecast.utils.download_tools import get_tigerforecast_dir
import os
import pandas as pd
import ast
from tigerforecast.utils.optimizers import *

static_features_to_use=[
        ('static:drain_area_log2', False),
        ('train:site:mean:USGS:discharge_mean', False),
        ('train:site:std:USGS:discharge_mean', False),
    ]

sequence_features_to_use=[
    ('sequence:GSMAP_MERGED:hourlyPrecipRate', False),
    ('sequence:GLDAS21:Tair_f_inst', True),
    ('sequence:AQUA_VI:NDVI', True)
]

label_feature_name = 'label:USGS:discharge_mean'
past_sequence_label_feature_name = 'sequence:USGS:discharge_mean'


def plot_loss_site_2231000(show_plot=True):
    tigerforecast_dir = get_tigerforecast_dir()
    data_path = os.path.join(tigerforecast_dir, 'data/loss_site_2231000.csv')
    df = pd.read_csv(data_path)

    for col in df:
        plt.plot(df[col], label=col)

    plt.title('loss at site 2231000')
    plt.legend(loc='upper left')
    plt.show(block=True)

def test_lstm(show_plot=True):
    n, m, l, h = 7, 1, 3, 10

    method = tigerforecast.method("LSTM")
    method.initialize(n, m, l, h, optimizer=Adam)
    loss = lambda pred, true: (pred - true)**2

    features = process_features()

    loss_series = []
    for i in range(len(features)-1):
        x = features[i]
        y_pred = method.predict(x)[0]
        y_true = features[i+1][-1]
        # print("y_pred: " + str(y_pred))
        # print("y_true: " + str(y_true))
        loss_series.append(loss(y_true, y_pred))
        method.update(y_true)
    # print(loss_series[-10:])

    # save results
    tigerforecast_dir = get_tigerforecast_dir()
    data_path = os.path.join(tigerforecast_dir, 'data/loss_site_2231000.csv')
    try:
        csv_input = pd.read_csv(data_path)
    except:
        csv_input = pd.DataFrame([])
    if 'LSTM_adam_loss' not in csv_input:
        csv_input['LSTM_adam_loss'] = loss_series
        csv_input.to_csv(data_path, index=False)
    
    if show_plot:
        plt.plot(loss_series)
        plt.title("LSTM-Adam loss on site 2231000")
        plt.show(block=True)
        plt.close()

def test_autoregressor(show_plot=True):
    # TEST ARMA
    df_site_2231000 = get_site_df()
    seq_len = 61
    rows = df_site_2231000.shape[0]

    label_series = []
    for i in range(int(rows/seq_len)):
        label_series += ast.literal_eval(df_site_2231000[past_sequence_label_feature_name].iloc[61*i])


    method = tigerforecast.method("AutoRegressor")
    #method.initialize(p, optimizer = ONS)
    method.initialize(p=3, n=1)
    loss = lambda y_true, y_pred: (y_true - y_pred)**2
 
    results = []
    running_average_loss = []

    for i in range(len(label_series)-1):
        cur_y_pred = method.predict(label_series[i])
        cur_y_true = label_series[i+1]
        cur_loss = loss(cur_y_true, cur_y_pred)
        method.update(cur_y_true)
        results.append(cur_loss)
        total_loss = np.sum(results[-100:])/100
        running_average_loss.append(total_loss)

    tigerforecast_dir = get_tigerforecast_dir()
    data_path = os.path.join(tigerforecast_dir, 'data/loss_site_2231000.csv')
    try:
        csv_input = pd.read_csv(data_path)
    except:
        csv_input = pd.DataFrame([])
    if 'ARMA_100_window_average_loss' not in csv_input:
        csv_input['ARMA_100_window_average_loss'] = running_average_loss
        csv_input.to_csv(data_path, index=False)

    if show_plot:
        plt.figure()
        plt.plot(results)
        plt.title("Autoregressive loss on " + past_sequence_label_feature_name)
        # plt.close()
        plt.figure()
        plt.plot(running_average_loss)
        plt.title("Autoregressive 100-window average loss on " + past_sequence_label_feature_name)
        plt.show(block=True)





def process_features():
    df_site_2231000 = get_site_df()
    seq_len = 61
    rows = df_site_2231000.shape[0]

    features = []
    for i in range(int(rows/seq_len)):
        all_cur_seq = []
        for (seq_feat, b) in sequence_features_to_use:
            cur_seq = ast.literal_eval(df_site_2231000[seq_feat].iloc[61*i])
            all_cur_seq.append(cur_seq)

        all_cur_seq.append(ast.literal_eval(df_site_2231000[past_sequence_label_feature_name].iloc[61*i]))

        for j in range(seq_len):
            next_feature = []
            for (stat_feat, c) in static_features_to_use:
                next_feature.append(ast.literal_eval(df_site_2231000[stat_feat].iloc[0])[0])
            for k in range(len(all_cur_seq)):
                next_feature.append(all_cur_seq[k][j])
            features.append(next_feature)

    return features 


def plot_flood_data_histograms(show_plot=True):
    df_site_2231000 = get_site_df()
    seq_len = 61
    rows = df_site_2231000.shape[0]
    '''
    for (seq_feat, b) in sequence_features_to_use:
        seq_feat_series = []
        for i in range(int(rows/seq_len)):
            seq_feat_series += ast.literal_eval(df_site_2231000[seq_feat].iloc[61*i])

        if show_plot:
            # print(seq_feat_series)
            plt.figure()
            plt.plot(seq_feat_series)
            plt.title(seq_feat)

    for (stat_feat, b) in static_features_to_use:
        stat_feat_series = []
        for i in range(rows):
            stat_feat_series += ast.literal_eval(df_site_2231000[stat_feat].iloc[i])

        if show_plot:
            # print(stat_feat_series)
            plt.figure()
            plt.plot(stat_feat_series)
            plt.title(stat_feat)

    label_series = []
    for i in range(int(rows/seq_len)):
        label_series += ast.literal_eval(df_site_2231000[past_sequence_label_feature_name].iloc[61*i])
    if show_plot:
        plt.figure()
        plt.plot(label_series)
        plt.title(past_sequence_label_feature_name)
    '''
    timestamp_series = []
    for i in range(20):
        stamp = df_site_2231000['label:timestamp'].iloc[i]
        print(stamp)
        timestamp_series += stamp
    print(timestamp_series[:20])

    if show_plot:
        plt.figure()
        plt.plot(timestamp_series)
        plt.title('label:timestamp')
    

    plt.show(block=True)



# https://cnsviewer.corp.google.com/cns/jn-d/home/floods/hydro_method/datasets/processed/full/
def get_site_df():
    
    tigerforecast_dir = get_tigerforecast_dir()
    data_path = os.path.join(tigerforecast_dir, 'data/FL_train.csv')
    df = pd.read_csv(data_path)
    # print(df['__site_id'].head())
    feature_list = []
    sequence_length = 61
    label_list = []
    
    print(df['label:timestamp'].iloc[0])
    print(type(df['label:timestamp'].iloc[0]))
    df_site_2231000 = df.loc[df['__site_id'] == 2376500]
    discharge_mean_0 = df_site_2231000[past_sequence_label_feature_name].iloc[0]
    rows = df_site_2231000.shape[0]

    # test if 61-seqs are redundant or disjoint
    '''
    for (seq_feat, b) in sequence_features_to_use:
        print("---------------------------------------")
        entry_0 = ast.literal_eval(df_site_2231000[seq_feat].iloc[0])
        entry_1 = ast.literal_eval(df_site_2231000[seq_feat].iloc[1])
        print(entry_0[:10])
        print(entry_1[:10])

    print("-------------------------------------------")
    label_0 = ast.literal_eval(df_site_2231000[past_sequence_label_feature_name].iloc[0])
    label_1 = ast.literal_eval(df_site_2231000[past_sequence_label_feature_name].iloc[1])
    print(label_0[:10])
    print(label_1[:10])'''

    # conclusion: seq features are redundant

    # test if static features are redundant
    '''
    for (stat_feat, b) in static_features_to_use:
        print("---------------------------------------")
        for i in range(int(rows/61)):
            entry_i = ast.literal_eval(df_site_2231000[stat_feat].iloc[61*i])[0]
            print(entry_i)'''
    # conclusion: static features are constant throughout time

    return df_site_2231000


if __name__=="__main__":
    # plot_loss_site_2231000()
    # test_lstm()
    # test_autoregressor()
    plot_flood_data_histograms(False)
    # get_site_df()