# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:50:06 2022

@author: Hugh

Select the best ML algorithm for Al alloys database
"""

import pandas as pd
import numpy as np
import concurrent.futures

from DataPreprocessing import RF_DataPreprocessing, SVR_DataPreprocessing
from FeatureEngineering import RF_FeatureEngineering
from TrainingFunction import NestedCV_concurrent, NN_Nested_CV

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor



'''Load and Process Dataset'''
# Define database path
path = 'dataset.xlsx'

# Read and process database in DataFrame - dataset dictionary = {property:[feature, target property],...}
data = pd.read_excel(path)
preprocessed_data = RF_DataPreprocessing(data)

# Find common low-VI features from all three properties & drop them from data
low_VI_list = RF_FeatureEngineering(preprocessed_data)
low_VI_dropped_data = SVR_DataPreprocessing(preprocessed_data, low_VI_list)



'''Model Selection for Regression'''
# Model list and default regressor
Model = ['RF', 'XGB', 'SVR.rbf']
Regressor = {'RF': RandomForestRegressor(),
             'XGB': XGBRegressor(),
             'SVR.rbf': SVR(kernel='rbf')}

# Model parameters for Grid Search
Model_para = {'RF': {'n_estimators': [50, 100, 200, 500],
                      'max_depth': [None, 5, 10, 20],
                      'min_samples_split': [2, 4, 8],
                      'min_samples_leaf': [1, 2, 4]},
              'XGB': {'n_estimators': [50, 100, 200, 500],
                      'max_depth': [2, 4, 8],
                      'learning_rate': [0.05, 0.1, 0.2, 0.5]},
              'SVR.rbf': {'C': [],
                          'gamma': []}}

# Model score dictionary and seed number and K fold
Model_score = {}
seed_num = 100
K = 5

# Loop for each model
for model in Model:
    print(model)
    # Define the default regressor & hyperparameter range & score dictionary
    regressor = Regressor[model]
    para_grid = Model_para[model]
    model_score = {}
    # Loop for each property
    for prop in low_VI_dropped_data:
        # Define the feature and label set & RMSE score
        X = low_VI_dropped_data[prop][0]
        y = low_VI_dropped_data[prop][1]
        test_RMSE, test_Acc = [], []
        # valid_RMSE, valid_Acc = [], []
        # Loop for each splitting seed in CV
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result_list = [executor.submit(NestedCV_concurrent, K, X, y, model, para_grid, regressor, i)
                           for i in range(seed_num)]
        for f in concurrent.futures.as_completed(result_list):
            scores, opt_para = f.result()
            test_RMSE.append(scores['test'][0])
            test_Acc.append(scores['test'][1])
        # Calculate the average and std of RMSE and Accuracy results for both testing and validation sets
        test_RMSE_mean, test_RMSE_std =  np.mean(test_RMSE), np.std(test_RMSE)
        test_Acc_mean, test_Acc_std = np.mean(test_Acc), np.std(test_Acc)
        # Store scores for each property under one model
        model_score[prop] = {'TRm': round(test_RMSE_mean, 4), 'TRs': round(test_RMSE_std, 4),
                             'TAm': round(test_Acc_mean, 4), 'TAs': round(test_Acc_std, 4),
                             'Optimal Parameters': opt_para}
        print('Model: ', model, 'Property: ', prop, 'RMSE: ', test_RMSE_mean, 'Optimal parameters: ', opt_para)
    Model_score[model] = model_score # Store RMSE score for each model

# Define NN hyperparameters and test for Neural Network
NN_para = {'neuron_list': [[256, 256], [512, 512], [1024, 1024], [256, 512], [512, 1024], [256, 256, 256], 
                            [512, 512, 512], [1024, 1024, 1024], [256, 512, 256], [512, 1024, 512]],
           'learning_rate': [0.0001, 0.0005, 0.001]}

model_score = {}
for prop in low_VI_dropped_data:
# for prop in low_VI_dropped_data:
    # Define feature and label of specific properties
    X = low_VI_dropped_data[prop][0]
    y = low_VI_dropped_data[prop][1]
    # Get the optimal RMSE on testing set
    scores, opt_para = NN_Nested_CV(X, y, NN_para, seed_num, K)
    model_score[prop] = {'TRm': round(scores['Test RMSE'][0], 4), 'TRs': round(scores['Test RMSE'][1], 4),
                         'TAm': round(scores['Test Acc'][0], 4), 'TAs': round(scores['Test Acc'][1], 4),
                         'Optimal Parameters': opt_para}
    
Model_score['NN'] = model_score

print(Model_score)