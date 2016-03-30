__author__ = 'barmalei4ik'

import pandas as pd
import matplotlib.pyplot as pyplot
import numpy as np
from activation_functions import sigmoid_function, tanh_function, linear_function,\
                                 LReLU_function, ReLU_function, elliot_function, symmetric_elliot_function, softmax_function
from cost_functions import sum_squared_error, cross_entropy_cost, exponential_cost, hellinger_distance, softmax_cross_entropy_cost
from learning_algorithms import backpropagation, scaled_conjugate_gradient, scipyoptimize, resilient_backpropagation
from neuralnet import NeuralNet
from tools import Instance

df = None

def create_ewma(period):
    pass

def create_macd():
    pass

def create_rsi():
    deltas = df["<CLOSE>"].diff()
    up, down = deltas.copy(), deltas.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = pd.ewma(up, 12)
    roll_down = pd.ewma(down.abs(), 12)
    df["rsi"] = 100.0 - (100.0 / (1.0 + roll_up / roll_down ))

def create_sok():
    pass

def create_sod():
    pass

def standardize(data):
    log_return = np.log(data).diff()
    std = pd.ewmstd(log_return, 10)
    ewma = pd.ewma(log_return, 10)
    data_standardized = 1 / (1 + np.exp((log_return - ewma) / std))

    return data_standardized


if __name__ == "__main__":

    df = pd.read_csv("data/EURCHF_day.csv", parse_dates=True, index_col=1)

    df["ema12"] = pd.ewma(df["<CLOSE>"], 12)
    df["ema26"] = pd.ewma(df["<CLOSE>"], 26)
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_smoothed"] = pd.ewma(df["macd"], 9)

    create_rsi()

    df["sok"] = (df["<CLOSE>"] - df["<LOW>"]) / (df["<HIGH>"] - df["<LOW>"])
    df["sod"] = pd.ewma(df["sok"], span = 3, min_periods = 2)

    df["close_stded"] = standardize(df["<CLOSE>"])
    df["ema12_stded"] = standardize(df["ema12"])
    df["ema26_stded"] = standardize(df["ema26"])
    df["macd_stded"] = standardize(df["macd_smoothed"])
    df["rsi_stded"] = standardize(df["rsi"])
    df["sod_stded"] = standardize(df["sod"])


    # Training sets
    training_one    = [ ]

    for index, row in df.iterrows():

        rsi = row["rsi_stded"]
        ema12 = row["ema12_stded"]
        ema26 = row["ema26_stded"]
        macd = row["macd_stded"]
        price = row["close_stded"]

        if not pd.isnull(rsi) and not pd.isnull(ema12) and not pd.isnull(ema26) and not pd.isnull(macd):
            #training_one.append( Instance([rsi, ema12, ema26, macd], [price]) )
            training_one.append( Instance([rsi, ema12], [price]) )

        if len(training_one) == 500:
            break

    settings = {
        # Required settings
        "cost_function"         : sum_squared_error,
        "n_inputs"              : 2,       # Number of network input signals
        "layers"                : [ (2, tanh_function), (1, sigmoid_function) ],
                                            # [ (number_of_neurons, activation_function) ]
                                            # The last pair in you list describes the number of output signals

        # Optional settings
        "weights_low"           : -0.1,     # Lower bound on initial weight range
        "weights_high"          : 0.1,      # Upper bound on initial weight range
        "save_trained_network"  : False,    # Whether to write the trained weights to disk

        "input_layer_dropout"   : 0.0,      # dropout fraction of the input layer
        "hidden_layer_dropout"  : 0.0,      # dropout fraction in all hidden layers
    }


    # initialize the neural network
    network = NeuralNet( settings )

    ## Train the network using Scaled Conjugate Gradient
    #scaled_conjugate_gradient(
    #        network,
    #        training_one,
    #        ERROR_LIMIT = 1e-4
    #    )

    # Train the network using backpropagation
    backpropagation(
            network,
            training_one,          # specify the training set
            #ERROR_LIMIT     = 1e-3, # define an acceptable error limit
            ERROR_LIMIT     = 0.2, # define an acceptable error limit
            #max_iterations  = 100, # continues until the error limit is reach if this argument is skipped

            # optional parameters
            learning_rate   = 0.03, # learning rate
            momentum_factor = 0.9, # momentum
         )

    network.print_test( training_one )




