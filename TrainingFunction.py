import numpy as np
import pandas as pd
import concurrent.futures

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from math import sqrt
from itertools import product

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam



'''Neural Network'''
def Build_NN(input_dimen, neuron_list, lerarning_rate):
    # Build NN model with Sequential
    NN_model = Sequential()
    NN_model.add(InputLayer(input_shape = (input_dimen, )))
    # Add fully-connected layers
    for i in range(len(neuron_list)):
        NN_model.add(Dense(neuron_list[i], activation = 'relu'))
        NN_model.add(Dropout(rate = 0.2))
    # Add output layer   
    NN_model.add(Dense(1))
    # Compile NN model with optimizer and loss and metrics
    optimizer = Adam(lerarning_rate)
    NN_model.compile(optimizer, loss = 'mean_squared_error', metrics = 'mean_squared_error')
    return NN_model

def NN_Nested_CV(feature, label, para_grid, seed_num, K = 5):
    '''
    Input: Feature & Label
           para_grid - search space of hyperparameters of NN model
           seed_num - Total seed number for random dataset splitting
    Output: 
    '''
    # Define the hyperparameters in para_grid
    neuron_dic = para_grid['neuron_list']
    lr_dic = para_grid['learning_rate']
    # Define lists to store scores
    test_RMSE_list, test_Acc_list = [], []
    # Define inner & outer KFold CV
    for seed in range(seed_num):
        outer_cv = KFold(n_splits = K, shuffle = True, random_state = seed)
        inner_cv = KFold(n_splits = K, shuffle = True, random_state = seed)
        for train, test in outer_cv.split(feature, label):
            # Define the training and testing dataset
            X_train, y_train = feature.iloc[train], label.iloc[train]
            X_test, y_test = feature.iloc[test], label.iloc[test]
            # Define the initial lowest RMSE value
            lowest_valid_RMSE = 100000
            # Loop over the para_grid
            for neuron_list in neuron_dic:
                for lr in lr_dic:
                    # Define the validation RMSE and NN model list
                    valid_RMSE_list, NN_model_list = [], []
                    for subtrain, valid in inner_cv.split(X_train, y_train):
                        # Define the subtrain and valid dataset
                        X_subtrain, y_subtrain = X_train.iloc[subtrain], y_train.iloc[subtrain]
                        X_valid, y_valid = X_train.iloc[valid], y_train.iloc[valid]
                        # Define the NN model and early stop criterion
                        NN_model = Build_NN(X_subtrain.shape[1], neuron_list, lr)
                        earlystopper = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 20, \
                                                     verbose = 0, mode = 'auto', restore_best_weights = True)
                        # Train the NN model
                        training = NN_model.fit(X_subtrain, y_subtrain, epochs = 1000, batch_size = 32, validation_split = 0, 
                                                validation_data = (X_valid, y_valid), callbacks = [earlystopper], verbose = 0)
                        # Store the RMSE result for validation set
                        valid_RMSE_list.append(sqrt(min(training.history['val_' + 'mean_squared_error'])))
                        # Store the trained NN models
                        NN_model_list.append(NN_model)
                    valid_RMSE_mean = sum(valid_RMSE_list) / len(valid_RMSE_list)
                    print(neuron_list, lr, valid_RMSE_mean)
                    if valid_RMSE_mean < lowest_valid_RMSE:
                        lowest_valid_RMSE = valid_RMSE_mean
                        best_model_list = NN_model_list
                        best_para = [neuron_list, lr]
            print(lowest_valid_RMSE, best_para)
            # Loop over best NN models and make prediciton
            for NN_model in best_model_list:
                test_pred = NN_model.predict(X_test)
                test_RMSE = mean_squared_error(y_test, test_pred, squared=False)
                test_Acc = r2_score(y_test, test_pred)
                test_RMSE_list.append(test_RMSE)
                test_Acc_list.append(test_Acc)
    Results = {'Test RMSE': [np.mean(test_RMSE_list), np.std(test_RMSE_list)],
               'Test Acc': [np.mean(test_Acc_list), np.std(test_Acc_list)]}
    return Results, best_para



'''Nested CV for concurrent process'''
# Define the score function RMSE for GridSearchCV
RMSE_scorer = make_scorer(mean_squared_error, squared = False, greater_is_better = False)
Acc_scorer = make_scorer(r2_score)
scorers = {'RMSE': RMSE_scorer, 'Acc': Acc_scorer}

def NestedCV_concurrent(K, feature, label, model, para_grid, regressor, seed):
    '''
    Grid Search for optimal hyperparameters of ML model based on KFold CV and apply
        to Testing set to check the generalizability
    
    Input: K - number for inner and outer K-fold CV
           Feature & Label
           model - ML model name
           para_grid - search space of hyperparameters
           regressor - ML model regressor
           seed - specific seed number for random dataset splitting
           
    Output: Result dictionary containing average and std of RMSE and Accuracy for both Testing and Validation sets
            Optimal Parameter for guidance of Parameter Grid selection
    '''
    # Define the RMSE & Accuracy for testing & validation set and inner & outer KFold CV
    valid_RMSE, valid_Acc = [], []
    test_RMSE, test_Acc = [], []
    outer_cv = KFold(n_splits = K, shuffle = True, random_state = seed)
    inner_cv = KFold(n_splits = K, shuffle = True, random_state = seed)
    # Loop for outer CV
    for train, test in outer_cv.split(feature, label):
        # SVR with rbf kernel need to tune gamma value
        if model == 'SVR.rbf': 
            # Define C and gamma parameter space
            C_step, gamma_step = 1000, 0.01
            para_grid['C'] = np.arange(100, 2000, C_step / 10)
            para_grid['gamma'] = np.arange(0.01, 0.1, gamma_step)
            # Reduce C step until step is 1
            while C_step != 1:
                # Loop over inner CV
                rgs = GridSearchCV(regressor, param_grid=para_grid, scoring=scorers, cv=inner_cv, refit='RMSE')
                rgs.fit(feature.iloc[train], label.iloc[train])
                # Update C range
                C_step = int(C_step / 10)
                opt_C = rgs.best_params_['C']
                C_lower, C_upper = opt_C - C_step + C_step / 10, opt_C + C_step
                if C_lower <= 0: C_lower = C_step # Avoid C being smaller than or equal to 0
                para_grid['C'] = np.arange(C_lower, C_upper, C_step / 10)
                # Update gamma range
                opt_gamma = rgs.best_params_['gamma']
                gamma_lower, gamma_upper = opt_gamma - gamma_step * 2, opt_gamma + gamma_step * 3
                if gamma_lower <= 0: gamma_lower = gamma_step # Avoid gamma being smaller than or equal to 0
                para_grid['gamma'] = [round(i,2) for i in np.arange(gamma_lower, gamma_upper, gamma_step)]
        # Other two kernels in SVR
        elif model == 'SVR.lin' or model == 'SVR.poly':
            C_step = 1000
            para_grid['C'] = np.arange(100, 2000, C_step/10)
            while C_step != 1:
                # Loop over inner CV
                C_step = int(C_step / 10)
                rgs = GridSearchCV(regressor, param_grid=para_grid, scoring=scorers, cv=inner_cv, refit='RMSE')
                rgs.fit(feature.iloc[train], label.iloc[train])
                # Update C range
                opt_C = rgs.best_params_['C']
                lower, upper = opt_C - C_step + C_step / 10, opt_C + C_step
                if lower <= 0: lower = C_step # Avoid C being smaller than or equal to 0
                para_grid['C'] = np.arange(lower, upper, C_step / 10)
        # Other models
        else:
            # Loop over inner CV
            rgs = GridSearchCV(regressor, param_grid=para_grid, scoring=scorers, cv = inner_cv, refit='RMSE')
            rgs.fit(feature.iloc[train], label.iloc[train])
        # Store RMSE and Accuracy of validation set
        valid_RMSE.append(-rgs.cv_results_['mean_test_RMSE'][rgs.best_index_])
        valid_Acc.append(rgs.cv_results_['mean_test_Acc'][rgs.best_index_])
        # Make prediction on testing set with optimal parameters and store RMSE and Aaccuray
        test_prediction = rgs.predict(feature.iloc[test])
        test_RMSE.append(mean_squared_error(label.iloc[test], test_prediction, squared=False))
        test_Acc.append(r2_score(label.iloc[test], test_prediction))
    
    result_dic = {'test': [np.mean(test_RMSE), np.mean(test_Acc)], 
                  'valid': [np.mean(valid_RMSE), np.mean(valid_Acc)]}
    
    return result_dic, rgs.best_params_



'''Self-defined grid search CV for ModelTraining'''
# Calculate average Acc between actual results and prediction under different seed splits
def Self_Kfold_CV(feature, label, model, param_keys, combination, seed_num, K=5):
    # Combine parameter and value into dictionary and set up the model
    params = dict(zip(param_keys, combination))
    model_instance = model(**params)
    # Get the accuracy results from different seeds
    RMSE_list = []
    Acc_list = []

    for i in range(seed_num):
        # Define total Accuracy for each seed
        RMSE = 0
        Acc = 0
        # Loop over K-fold to sum up Accuracy
        kf = KFold(n_splits = K, shuffle = True, random_state = i)
        for train, val in kf.split(feature, label):
            model_instance.fit(feature.iloc[train], label.iloc[train])
            prediction = model_instance.predict(feature.iloc[val])
            RMSE += mean_squared_error(label.iloc[val], prediction, squared=False)
            Acc += model_instance.score(feature.iloc[val], label.iloc[val])
        # Append Accurcy of each seed to the list
        RMSE_list.append(RMSE / K)
        Acc_list.append(Acc / K)
    # Calculate the mean and std of Accuracy and RMSE
    RMSE_mean = round(np.mean(RMSE_list), 4)
    RMSE_std = round(np.std(RMSE_list), 4)
    Acc_mean = round(np.mean(Acc_list)*100, 4)
    Acc_std = round(np.std(Acc_list)*100, 4)

    return RMSE_mean, RMSE_std, Acc_mean, Acc_std, params

# Search for the optimal parameters based on the average Accuracy over multiple seeds
def Self_GridSearchCV(X, y, model, regressor, param_grid, seed_num):
    # Define the score dictionary to store parameters and corresponding scores
    Score_dic = {}
    if model == 'SVR.rbf': # Determine opt param for SVR.rbf model
        C_step, gamma_step = 100, 0.01
        param_grid['C'] = [round(i) for i in np.arange(100, 800, C_step)]
        param_grid['gamma'] = [round(i,2) for i in np.arange(0.02, 0.1, gamma_step)]
        param_keys = param_grid.keys() # get the parameter of regressor
        while C_step + 0.1 > 1: # Loop the optimization until C step less than 10
            # Grid search over the parameter space
            param_values = param_grid.values()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result_list = [executor.submit(Self_Kfold_CV, X, y, regressor, param_keys, combination, 
                                               seed_num) for combination in product(*param_values)]
            for f in concurrent.futures.as_completed(result_list):
                RMSE_mean, RMSE_std, Acc_mean, Acc_std, params = f.result()
                Score_dic[tuple(params.values())] = [RMSE_mean, RMSE_std, Acc_mean, Acc_std, params.keys()]
            # Get the hyperparameters with lowest RMSE for validation set in KFold CV   
            opt_para = min(Score_dic, key = Score_dic.get) # Get the optimal parameter
            opt_C, opt_gamma = opt_para[1], opt_para[2] # Get the optimal C & gamma
            # Update C range
            C_step = int(C_step / 10)
            C_lower, C_upper = opt_C - C_step * 9, opt_C + C_step * 10
            if C_step != 0:
                if C_lower <= 0: C_lower = C_step # Avoid C being smaller than or equal to 0
                param_grid['C'] = [round(i) for i in np.arange(C_lower, C_upper, C_step)]
                # Update gamma range
                gamma_lower, gamma_upper = opt_gamma - gamma_step * 2, opt_gamma + gamma_step * 3
                if gamma_lower <= 0: gamma_lower = gamma_step # Make sure gamma larger than 0
                param_grid['gamma'] = [round(i,2) for i in np.arange(gamma_lower,gamma_upper,gamma_step)]
    else: # Determine opt param for other models
        # Get parameters and corresponding values
        param_keys = param_grid.keys()
        param_values = param_grid.values()
        # Loop over all combinations of parameter values
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result_list = [executor.submit(Self_Kfold_CV, X, y, regressor, param_keys, combination, 
                                           seed_num) for combination in product(*param_values)]
        for f in concurrent.futures.as_completed(result_list):
            RMSE_mean, RMSE_std, Acc_mean, Acc_std, params = f.result()
            Score_dic[tuple(params.values())] = [RMSE_mean, RMSE_std, Acc_mean, Acc_std, params.keys()]
        opt_para = min(Score_dic, key = Score_dic.get)
    
    # return [mean, std], param_keys, optimal param_values
    return Score_dic[opt_para][0:4], Score_dic[opt_para][4], opt_para



'''Calculate RMSE and/or accuracy for KFold CV method'''
# Calculate avergae RMSE between actual results and prediction under different seed splits
def Kfold_CV(feature, label, model, seed, K=5):
    '''
    Input: Feature & Label
           model - ML model with predetermined hyperparameters
           seed - specific seed number for random dataset splitting
           K - number of split in K-fold CV
           
    Output: average RMSE between actual and predicted results in validation set
    '''
    # Define total RMSE
    RMSE = 0
    Acc = 0   
    
    # Loop over K-fold to sum up RMSE
    kf = KFold(n_splits = K, shuffle = True, random_state = seed)

    for train, val in kf.split(feature, label):
        model.fit(feature.iloc[train], label.iloc[train])
        prediction = model.predict(feature.iloc[val])
        RMSE += mean_squared_error(label.iloc[val], prediction, squared=False)
        Acc += model.score(feature.iloc[val], label.iloc[val])
        
    return RMSE / K, Acc / K

# Calculate RMSE between actual results and average prediction under different seed splits
def Kfold_CV_AveragePrediction(feature, label, model, seed_num, K=5):
    '''
    Input: Feature & Label
           ML model with predetermined hyperparameters
           Total seed number for random dataset splitting
           K number for K-fold CV
           
    Output: RMSE between actual results and average prediction
    '''
    
    # Define a df with zero to store all predictions in validation set from different seed
    total_pred_df = pd.DataFrame(np.zeros((len(label),1)))
    
    # Loop over all seeds and K-fold to sum up RMSE
    for seed in range(seed_num):
        # Define the K fold splits
        kf = KFold(n_splits = K, shuffle = True, random_state = seed)
        # Define index list and prediction list to sort prediction of validation set in order
        index_list = []
        prediction_list = [] 
        
        # Feed different train and validation set into regression model for K times
        for train, val in kf.split(feature, label):
            model.fit(feature.iloc[train], label.iloc[train])
            prediction = model.predict(feature.iloc[val])  
            # Extend the list of index and prediction
            index_list.extend(val.tolist())
            prediction_list.extend(prediction.tolist())
            
        # Store prediction in DataFrame with original index of label data
        prediction_df = pd.DataFrame(prediction_list, index = index_list)
        # Return sorted prediction list by index
        total_pred_df += prediction_df.sort_index()
        
    # Divide total prediction by number of seeds to get mean prediction
    pred_df = total_pred_df/seed_num
    RMSE = mean_squared_error(label, pred_df, squared = False)
    
    return RMSE