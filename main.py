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

import pandas as pd
import custom_functions
import sys

################## Make csv out of randomly created wallets ##################

# import secrets
# import custom_functions

# ls = []
# how_many_wallets = 10
#
# n = 0
# while n < how_many_wallets:
#
#     list_of_values = None
#     while list_of_values is None:
#         try:
#             bits = secrets.randbits(256)
#             list_of_values = custom_functions.make_my_wallet(bits)
#         except:
#             pass
#
#     ls.append(list_of_values)
#     n += 1
#
# df_of_wallets = custom_functions.make_df_wallets_and_export(ls)


################## Load saved wallets ##################

location = 'CRYPTOKEYS.csv'
df = pd.read_csv(location)

################## Data Preprocessing to apply DL model ##################
df_copy = custom_functions.make_file(df)

for each in list(df_copy.columns[:-1]):
    df_copy[each] = df_copy.apply(lambda row: custom_functions.map_out(row[each]), axis=1)

jumps = 5

df_copy['DEC_key_2'] = df_copy.apply(lambda row : custom_functions.split_dec_vals(row['DEC_key'],jumps), axis = 1)
list_val = list(range(100,100+len(df_copy['DEC_key_2'].tolist()[0])))
df_copy[list_val] = pd.DataFrame(df_copy.DEC_key_2.tolist(), index= df_copy.index)
df_copy.dropna(inplace=True)
df_copy.reset_index(drop=True, inplace=True)
df_copy = df_copy.drop(columns = ['DEC_key', 'DEC_key_2'])
df_copy = df_copy.astype('int64')

# Creating train, test and validation splits
split_ratio = 0.8
df_train, df_test, data_validation = custom_functions.train_test_split(df_copy, split_ratio)

x_variables = df_copy.columns[0:34].tolist()
# x_variables = df_train.columns[5:10].tolist()
y_variables = 103

x_train = df_train[x_variables].values
y_train = df_train[y_variables].values

x_test = df_test[x_variables].values
y_test = df_test[y_variables].values

x_val = data_validation[x_variables].values
y_val = data_validation[y_variables].values

# Setting up model
epochs = 500
batchsize = 100
use_validation = 'y'
validation_split = 0.2
patience = 50

# Choose and make model
which_model_to_make = input('Choose between LSTM (1) and CNN 1d (2). Type 1 or 2: ')

if which_model_to_make == '1':
    x_test, x_train, x_val = custom_functions.reshape_for_LSTM(x_test, x_train, x_val)
    model, es, mc = custom_functions.LSTM_MODEL(x_train, patience)
    plot_location_TV = 'All_Exports/03_Model_Logs/Plots/01_LSTM_Train_vs_Validation.png'
    plot_location_P = 'All_Exports/03_Model_Logs/Plots/02_LSTM_Prediction.png'
    model_name = 'LSTM'
elif which_model_to_make == '2':
    x_test, x_train, x_val = custom_functions.reshape_for_CNN1D(x_test, x_train, x_val)
    model, es, mc = custom_functions.build_conv1D_model(x_train, patience)
    plot_location_TV = 'All_Exports/03_Model_Logs/Plots/01_CNN1D_Train_vs_Validation.png'
    plot_location_P = 'All_Exports/03_Model_Logs/Plots/02_CNN1D_Prediction.png'
    model_name = 'CNN1D'
else:
    print('Invalid Input')
    sys.exit()

# Train Model
model, history = custom_functions.run_model(x_train, y_train,
                                            model, epochs,
                                            use_validation, validation_split,
                                            batchsize, es, mc)

# Exporting plots to disk
custom_functions.make_and_export_plots(history, model, x_val, y_val, plot_location_TV, plot_location_P, model_name)