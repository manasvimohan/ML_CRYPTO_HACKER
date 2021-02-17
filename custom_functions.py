'''
-----------------------------------------------------------------------
Created by Manasvi Mohan Sharma on 17/02/21 (dd/mm/yy)
Project Name: ML_CRYPTO_HACKER | File Name: main.py
IDE: PyCharm | Python Version: 3.8
-----------------------------------------------------------------------
                                       _
                                      (_)
 _ __ ___   __ _ _ __   __ _ _____   ___
| '_ ` _ \ / _` | '_ \ / _` / __\ \ / / |
| | | | | | (_| | | | | (_| \__ \\ V /| |
|_| |_| |_|\__,_|_| |_|\__,_|___/ \_/ |_|

GitHub:   https://github.com/manasvimohan
Linkedin: https://www.linkedin.com/in/manasvi-mohan-sharma-119375168/
Website:  https://www.manasvi.co.in/
-----------------------------------------------------------------------
Project Information:
This is a fun project to create Deep Learning models using tensorflow
to predict private key of a bitcoin wallet from the public address of
the wallet. Does not work, however is worth a try in today's day and age.

Such projects make you thing about data engineering and pre processing
and how this impossible task can be tackled and different ways encrypted
data can be hacked.

About this file:
This is the main file.
-----------------------------------------------------------------------
'''

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
fig_breadth = 14
fig_height = fig_breadth/2
plt.style.use('seaborn-pastel')


def str_to_list(row_val):
    final = list(row_val)
    return final

def make_file(df):
    df_copy = df[['Public_Key','DEC_key']].copy()
    
    total = len(df_copy['Public_Key'][0])

    df_copy["Public_Key"] = df_copy.apply(lambda row : str_to_list(row['Public_Key']), axis = 1) 

    list_val = list(range(total))
    df_copy[list_val] = pd.DataFrame(df_copy.Public_Key.tolist(), index= df_copy.index)

    df_copy = df_copy.drop(columns='Public_Key')

    cols = df_copy.columns.tolist()
    cols =  cols[-total:]+cols[0:1]
    
    df_copy = df_copy[cols] 
    df_copy.dropna(inplace=True)
    df_copy.reset_index(drop=True,inplace=True)

    return df_copy

def map_out(val):
    
    alphabet = list("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
    mapping = list(range(len(alphabet)))
    
    for a, b in zip(alphabet, mapping):
        
        if val == a:
            output = b
            
    return int(output)

def split_dec_vals(val, jumps):
    start = 0
    end = 77
    ls = []
    
    while start<end:
        try:
            yo  = int(val[start:start+jumps])
            start += jumps
            ls.append(yo)
        except:
            start += jumps
            pass
        
    return ls
    
def split_dec_vals_str(val, jumps):
    start = 0
    end = 77
    ls = []
    
    while start<end:
        try:
            yo  = val[start:start+jumps]
            start += jumps
            ls.append(yo)
        except:
            start += jumps
            pass
        
    return ls



def train_test_split(data, split_ratio):

    data_validation = data[-200:].copy() # Last 200 rows for validation
    data_test_train = data[:-200].copy() # rows zero to end minus 200 choosen in validation

    # Splitting randomly
    selection = np.random.rand(len(data_test_train)) < split_ratio
    df_train, df_test = data_test_train[selection], data_test_train[~selection]

    return df_train, df_test, data_validation

##### CNN1d Functions #####
def reshape_for_CNN1D(x_test, x_train, x_val):

    input_dimension = 1

    sample_size = x_test.shape[0]
    time_steps = x_test.shape[1]
    x_test = x_test.reshape(sample_size, time_steps, input_dimension)

    sample_size = x_train.shape[0]
    time_steps = x_train.shape[1]
    x_train = x_train.reshape(sample_size, time_steps, input_dimension)

    sample_size = x_val.shape[0]
    time_steps = x_val.shape[1]
    x_val = x_val.reshape(sample_size, time_steps, input_dimension)

    return x_test, x_train, x_val
def build_conv1D_model(x_train, patience):
    n_timesteps = x_train.shape[1]
    n_features = x_train.shape[2]

    model = Sequential()
    model.add(Input(shape=(n_timesteps, n_features)))

    model.add(Conv1D(filters=64, kernel_size=7, activation='relu', name="Conv1D_1",  input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', name="Conv1D_2"))
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu', name="Conv1D_3"))
    model.add(MaxPooling1D(2, padding='same', name="MaxPooling1D"))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', name="Dense_1"))
    model.add(Dense(n_features, name="Dense_2"))

    optimizer = RMSprop(0.001)
    loss = 'mean_squared_error'
    metrics = ['accuracy', 'mae']

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    print(model.summary())

    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       patience=patience)

    best_model_save_location = 'All_Exports/02_Exported_Models/Best_Models/best_model_cnn' + ".h5"

    mc = ModelCheckpoint(best_model_save_location,
                         monitor='val_accuracy',
                         mode='max',
                         verbose=0,
                         save_best_only=True)

    print(model.summary())

    return model, es, mc
###########################

##### LSTM Functions #####
def reshape_for_LSTM(x_test, x_train, x_val):

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

    return x_test, x_train, x_val
def LSTM_MODEL(x_train, patience):
    model = Sequential()
    n_batchsize = x_train.shape[1]
    n_features = 1
    model.add(Input(shape=(n_batchsize, n_features)))

    model.add(LSTM(60, return_sequences=True))
    model.add(LSTM(60, return_sequences=False))
    model.add(Dense(30))
    model.add(Dense(1))
    optimizer = 'adam'
    loss = 'mean_squared_error'
    metrics = ['accuracy', 'mae']
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       patience=patience)

    best_model_save_location = 'All_Exports/02_Exported_Models/Best_Models/best_model_lstm' + ".h5"

    mc = ModelCheckpoint(best_model_save_location,
                         monitor='val_accuracy',
                         mode='max',
                         verbose=0,
                         save_best_only=True)

    print(model.summary())

    return model, es, mc
##########################

##### Running Keras Model Function #####
def run_model(x_train, y_train, model, epochs, use_validation, validation_split, batchsize, es, mc):
    if use_validation == 'n':
        history = model.fit(x_train, y_train,
                            batch_size=batchsize, epochs=epochs, verbose =1,
                            shuffle=True,
                            callbacks = [es,mc])
    elif use_validation == 'y':
        history = model.fit(x_train, y_train,
                            batch_size=batchsize, epochs=epochs, verbose =1,
                            validation_split=validation_split, shuffle=True,
                            callbacks = [es,mc])
    else:
        print('Enter y or n for use_validation')
    return model, history
########################################

def make_and_export_plots(history, model, x_val, y_val, plot_location_TV, plot_location_P, model_name):
    # TRAIN VAL PLOT
    plt_train_vs_val = plt
    plt_train_vs_val.figure(figsize=(fig_breadth, fig_height))
    plt_train_vs_val.plot(history.history['loss'])
    plt_train_vs_val.plot(history.history['val_loss'])
    plt_train_vs_val.title('Model train vs Validation loss - '+ model_name)
    plt_train_vs_val.ylabel('Loss')
    plt_train_vs_val.xlabel('Epoch')
    plt_train_vs_val.legend(['Train', 'Validation'], loc='upper right')
    plt_train_vs_val.savefig(plot_location_TV)
    del plt_train_vs_val

    # PREDICTION AND PLOT
    predictions = model.predict(x_val)
    rmse = np.sqrt(np.mean(((predictions - y_val) ** 2)))
    rmse = int(round(rmse, 0))
    print('RMSE is {}'.format(rmse))

    plt_prediction = plt
    plt_prediction.figure(figsize=(fig_breadth, fig_height))
    plt_prediction.plot(predictions)
    plt_prediction.plot(y_val)
    plt_prediction.title('Predictions on validation set - '+ model_name + ' - RMSE - '+str(rmse))
    plt_prediction.ylabel('ACTUAL')
    plt_prediction.xlabel('Observations')
    plt_prediction.legend(['prediction', 'validation'], loc='upper right')
    plt_prediction.savefig(plot_location_P)
    del plt_prediction

import hashlib

def make_df_wallets_and_export(ls):
    columns = ['DEC_key','HEX_key', 'Public_Key', 'Private_key']

    df_of_wallets = pd.DataFrame(ls, columns = columns)

    # export_wallets_loc = 'hmmmmmm'
    # df_of_wallets.to_csv(export_wallets_loc+'.csv')
    return df_of_wallets


def base58(address_hex):
    
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    b58_string = ""
    # Get the number of leading zeros
    leading_zeros = len(address_hex) - len(address_hex.lstrip('0'))
    # Convert hex to decimal
    address_int = int(address_hex, 16)
    # Append digits to the start of string
    while address_int > 0:
        digit = address_int % 58
        digit_char = alphabet[digit]
        b58_string = digit_char + b58_string
        address_int //= 58
    # Add ‘1’ for each 2 leading zeros
    ones = leading_zeros // 2
    for one in range(ones):
        b58_string = '1' + b58_string
    return b58_string

def MAKE_WIF(private_key):
    
    import codecs
    
    PK0 = private_key
    PK1 = '80'+ PK0
    PK2 = hashlib.sha256(codecs.decode(PK1, 'hex'))
    PK3 = hashlib.sha256(PK2.digest())
    checksum = codecs.encode(PK3.digest(), 'hex')[0:8]
    PK4 = PK1 + str(checksum)[2:10]
    
    address_hex = PK4

    alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    b58_string = ''
    # Get the number of leading zeros
    leading_zeros = len(address_hex) - len(address_hex.lstrip('0'))
    # Convert hex to decimal
    address_int = int(address_hex, 16)
    # Append digits to the start of string
    while address_int > 0:
        digit = address_int % 58
        digit_char = alphabet[digit]
        b58_string = digit_char + b58_string
        address_int //= 58
    # Add ‘1’ for each 2 leading zeros
    ones = leading_zeros // 2
    for one in range(ones):
        b58_string = '1' + b58_string
    return b58_string

def WIF_TO_HEX_KEY(WIF):
    
    import base58
    import binascii

    private_key_WIF = WIF
    first_encode = base58.b58decode(private_key_WIF)
    private_key_full = binascii.hexlify(first_encode)
    private_key = private_key_full[2:-8]
    return private_key.decode('ASCII')

def printing_function(bits, private_key,Final_Address_base58,WIF):
    print('1) Your choice of number: {}'.format(bits))
    print('2) Your HEX private key: {}'.format(private_key))
    print('3) Your public Address: {}'.format(Final_Address_base58))
    print('4) Your Wallet Import Format Private Key: {}'.format(WIF))
    

def make_my_wallet(bits):

    bits_hex = hex(bits)
    private_key = bits_hex[2:]

    sha256_version = hashlib.sha256(private_key.encode()).hexdigest()

    ripemd160_version = hashlib.new('ripemd160')
    ripemd160_version.update(sha256_version.encode())
    ripemd160_version.hexdigest()

    mainnet = '00'+ripemd160_version.hexdigest()

    double_sha256 = hashlib.sha256(mainnet.encode()).hexdigest()
    double_sha256_final = hashlib.sha256(double_sha256.encode()).hexdigest()

    checksum = double_sha256_final[0:8]

    Final_Address = mainnet + checksum

    Final_Address_base58 = base58(Final_Address)

    WIF = MAKE_WIF(private_key)
    private_hex_key_from_wif = WIF_TO_HEX_KEY(WIF)
    
    return [bits, private_key, Final_Address_base58, WIF]
