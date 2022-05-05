# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 09:32:04 2022

@author: jo_as
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras.backend as K
from ClassRNN import RNN
import matplotlib.pyplot as plt


def smooth_third_deviation(y): #assumes burstes are distributet randomly and two are not found in a row
    E = np.sum(y)/len(y)
    var = np.sum((y - E)**2)
    for i in range(len(y)):
        if y[i] > E + 3*np.sqrt(var) or y[i] < E - 3*np.sqrt(var):
            if i == 0:
                y[0] = y[1]
            elif i == len(y)-1:
                y[-1] = y[-2]
            else:
                y[i] = E
    return y
            
def smooth_filter(y): #assumes burstes are distributet randomly and two are not found in a row
    count = 0
    for i in range(len(y)):
        if i < 5: 
            zum = (np.sum(y[0:i+5])  - y[i])/ (6 + i)
        elif i > len(y) - 5:
            zum = (np.sum(y[i-5:-1]) - y[i]) / (6 + len(y) - i)
        else:
            zum = (np.sum(y[i-5:i+5]) - y[i]) / 11
        if abs(y[i]) > 2*abs(zum):
            y[i] = zum
            count += 1
    print(f"{count/len(y)} datapoints smoothed in clamming")
    return y


def add_lag_features(data, mean = False, all_lags = True, naive = False):
    if not naive:
        # print(data.head())
        data["y_prev"] = data["y"].shift(1)
        # print(data.head())
        if all_lags:
            data["y_prev_day"] = data["y"].shift(12*24)
            # print(data.head())
            data["y_prev_week"] = data["y"].shift(12*24*7)
            # print(data.head())
            data = data.drop(data.index[0:12*24*7])
            # print(data.head())
        else:
            data = data.drop(data.index[0])

        
    
    
    
    y_index = data.columns.get_loc("y")
    # print(y_index)
    cols = data.columns.tolist()
    # print(cols)
    if y_index == len(data):
        print("y feature already in last column")
    else:
        cols = cols[:y_index] + cols[y_index + 1:] + cols[y_index:y_index + 1]
        # print(cols)
        data = data[cols]
        print("final head after adding features are \n" , data.head())
        if naive:
            print("naive model called")
    
    
    return data

def add_time_features(data):
    df_hour = pd.to_datetime(data["start_time"]).dt.hour
    print(type(data))
    print(type(df_hour))
    df_hour[0]
    df_day = pd.to_datetime(data["start_time"]).dt.day_name()
    df_month = pd.to_datetime(data["start_time"]).dt.month_name()
    print((df_hour[3]))
    for i in range(len(data)):
        if 0 == df_hour[i]:
            print("it is")
            break
        if i == len(data) - 1:
            print("it is not")
    # dayCategoricals = pd.get_dummies(df_hour)
    
    data["isNight"] = df_hour.isin(range(0,6))
    data["isMorning"] = df_hour.isin(range(12,6))
    data["isDay"] = df_hour.isin(range(12,18))
    data["isEvening"] = df_hour.isin(range(18,24))
    data["isWeekend"] = df_day.isin(["Saturday", "Sunday"])
    data["isWinter"] = df_month.isin(["December", "January", "February"])
    data["isSpring"] = df_month.isin(["March", "April", "May"])
    data["isSummer"] = df_month.isin(["June", "July", "August"])
    data["isFall"] = df_month.isin(["September", "October", "November"])
    
    # print(data.head())
    
    return data

def clamming(data, alpha):
    lower = data["y"].quantile(alpha)#remove 1% of data where most gets removed from upper as plot shows more outlier data
    upper = data["y"].quantile(1 - (0.01 - alpha))
    data["y"].clip(lower, upper, inplace = True)
    return data

def get_imbalance_error(data):
    df_minute = pd.to_datetime(data["start_time"]).dt.minute
    # print(df_minute.head)
    # for i in range(len(data)):
    #     if 0 == df_minute[i]:
    #         print("it is")
    #         break
    #     if i == len(data) - 1:
    #         print("it is not")
    data["o'clock"] = df_minute.isin([0])
    # print(data["o'clock"][10:])
    x = np.asarray(data.index[data["o'clock"] == True].tolist())
    # print(type(x))
    y = np.zeros(len(x))
    for count, xi in enumerate(x):
        y[count] = data["total"][xi]
    x_nodes = np.zeros(len(x)+1)
    y_vals = np.zeros(len(x)+1)
    x_mid = (x[0:-1] + x[1:])/2
    y_mid = y[0:-1]
    x_nodes[1:-1] = x_mid
    x_nodes[0] = 0
    x_nodes[-1] = len(x) - 1#at last datapoint in data
    y_vals[1:-1] = y_mid
    y_vals[0] = data["total"][0]
    y_vals[-1] = data["total"][len(data)-1]
    
    f = interp1d(x_mid, y_mid, kind = "cubic", fill_value = "extrapolate")
    x_new = np.arange(0,len(data))
    y_new = f(x_new)
    error = np.asarray(data["total"]) - y_new
    fig, ax = plt.subplots((2))
    ax[0].set_title("first 100 points")
    ax[0].plot(x_new[:100],data["total"][:100], label = "real")
    ax[0].plot(x_new[:100], y_new[:100], label = "interpol")
    ax[0].legend()
    ax[1].plot(x_new[:100], error[:100], label = "error")
    ax[1].legend()
    
    fig, ax = plt.subplots((2))
    N = len(x_new)//2
    ax[0].set_title("middle 100 points")
    ax[0].plot(x_new[N:N + 100],data["total"][N:N + 100], label = "real")
    ax[0].plot(x_new[N:N + 100], y_new[N:N + 100], label = "interpol")
    ax[0].legend()
    ax[1].plot(x_new[N:N + 100], error[N:N + 100], label = "error")
    ax[1].legend()
    
    fig, ax = plt.subplots((2))
    ax[0].set_title("last 100 points")
    ax[0].plot(x_new[-100:],data["total"][-100:], label = "real")
    ax[0].plot(x_new[-100:], y_new[-100:], label = "interpol")
    ax[0].legend()
    ax[1].plot(x_new[-100:], error[-100:], label = "error")
    ax[1].legend()
    
    return error

def create_dataset(data_train, data_val, n_seq, alpha = 0.005, add_features = [], alt = False):
    ###Altering
    
    if alt:
        training_error = get_imbalance_error(data_train)
        val_error = get_imbalance_error(data_val)
        print(f"training error type is {type(training_error)}")
        print(training_error[100:110])
        data_train["y"] = data_train["y"] - training_error
        data_val["y"] = data_val["y"] - val_error
        
    ###Clamming
    data_train = clamming(data_train, alpha)
    data_val = clamming(data_train, alpha)
    ###Adding features
    if "time" in add_features:
        data_train = add_time_features(data_train)
        data_val = add_time_features(data_val)
    else:
        print("no time featurs added")
        
    if "lag" in add_features:
        data_train = add_lag_features(data_train, all_lags = True)
        data_val = add_lag_features(data_val, all_lags = True)
    else:
        if "naive" in add_features:
            data_train = add_lag_features(data_train, naive = True)
            data_val = add_lag_features(data_val, naive = True)
            print("no features added at all, including y_prev")
        else:
            data_train = add_lag_features(data_train, all_lags = False)
            data_val = add_lag_features(data_val, all_lags = False)
        print("no additional lag features added")
    data_train = data_train.drop(["start_time"], axis=1)
    data_val = data_val.drop(["start_time"], axis=1)
    
    feature_list = data_train.columns

    ###Normalizing 
    # data, norm = preprocessing.normalize(np.asarray(data.drop(["start_time"], axis=1)).astype(np.float32), return_norm = True, axis = 0)

    scaler = preprocessing.StandardScaler()
    data_train = scaler.fit_transform(data_train)
    data_val = scaler.transform(data_val)
    
    #y_data = smooth_third_deviation(y_data)
    x_data_train = data_train[:,:-1]
    y_data_train = data_train[:,[-1]]
    x_data_val = data_val[:,:-1]
    y_data_val = data_val[:,[-1]]
    
    
    x_list_train, y_list_train = [], []
    x_list_val, y_list_val = [], []
    
    for i in range(len(x_data_train) - n_seq - 1):
        x_list_train.append(x_data_train[i:(i+n_seq)]) #n_seq long inputs up to but not included i + n_seq. This includes y_prevs up tothe 5 minutes before
        y_list_train.append(y_data_train[(i+n_seq)]) #The imbalance y estimated at t = t_{i+n_seq}
    for i in range(len(x_data_val) - n_seq - 1):
        x_list_val.append(x_data_val[i:(i+n_seq)])
        y_list_val.append(y_data_val[(i+n_seq)])
    if alt:
        return np.array(x_list_train), np.array(y_list_train), np.array(x_list_val), np.array(y_list_val), feature_list, training_error, val_error
    return np.array(x_list_train), np.array(y_list_train), np.array(x_list_val), np.array(y_list_val), feature_list

# struct_imb(df_train,-80,-1)

# def struct_imb2(data, x0,x1):