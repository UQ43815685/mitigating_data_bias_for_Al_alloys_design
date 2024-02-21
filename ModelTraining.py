# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:02:31 2022

@author: Hugh

Best ML model is selected in the ModelSelection module for each property. 
This module select the optimal parameters based on 5-fold CV on entire data set with 10 random state seed.
It is 50 RMSE score for each hyperparameter set (5-fold * 10 seed).
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn import svm
from DataPreprocessing import RF_DataPreprocessing, SVR_DataPreprocessing
from FeatureEngineering import RF_FeatureEngineering
from TrainingFunction import Kfold_CV, Self_GridSearchCV



'''Define dataset and process dataset'''
# Define database path
path = 'dataset.xlsx'

# Read and process database in DataFrame - dataset dictionary = {property:[feature, target property],...}
data = pd.read_excel(path)
preprocessed_data = RF_DataPreprocessing(data)
# Find common low-VI features from all three properties & drop them from data
low_VI_list = RF_FeatureEngineering(preprocessed_data)
low_VI_dropped_data = SVR_DataPreprocessing(preprocessed_data, low_VI_list)

'''Training SVR and XGB model for all three mechanical properties'''
seed_num = 100

Model = {'YTS': ['SVR.rbf', svm.SVR, {'kernel': ['rbf'], 'C': [], 'gamma': []}],
         'UTS': ['SVR.rbf', svm.SVR, {'kernel': ['rbf'], 'C': [], 'gamma': []}],
         'EL': ['XGB',xgb.XGBRegressor, {'n_estimators': [500, 1000, 2000],
                                          'max_depth': [2, 4, 8],
                                          'learning_rate': [0.05, 0.1, 0.2]}]}

for prop in low_VI_dropped_data:
    # Define the feature and label set
    X = low_VI_dropped_data[prop][0]
    y = low_VI_dropped_data[prop][1]
    # Define the model used and parameter grid
    model = Model[prop][0]
    regressor = Model[prop][1]
    params_grid = Model[prop][2]
    # Start grid search for optimal parameters
    Perf, param_keys, param_values = Self_GridSearchCV(X, y, model, regressor, params_grid, seed_num)
    params = dict(zip(param_keys, param_values))
    print(prop, 'Optimal parameters:',params, '/nRMSE: ', Perf[0], 'std: ', Perf[1], 
          'Accuracy: ', Perf[2], 'std: ', Perf[3])