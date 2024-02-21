import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import preprocessing, svm
from math import log, sqrt



'''Calculate fitness of population'''
def Pred_DataProcessing(dataset, dataset_pred):
    """
    Preprocess training and prediction dataset for GA
    Return: 
        dataset dictionary = {property:[feature, target property],...}
        prediction dictionary = {property: feature}
    """
    
    # Preprocess training and prediction dataset
    mech_props = ['YTS','UTS','EL']
    drop_list = ['Alloy','YTS','UTS','EL','Al (min.)','Reference','Designation']
    common_low_VIs = ['Ga', 'Bi', 'B', 'Pb', 'Ni', 'Be', 'V']
    dataset_dic = {}
    pred_dic = {}
    
    # Convert prediction dataset to df with dataset column name and dtypes
    df_pred=pd.DataFrame(dataset_pred, columns=dataset.drop(drop_list+common_low_VIs,axis=1).columns).convert_dtypes(convert_string=False)
    for col in df_pred:
        if df_pred[col].dtype == 'object': break
        else: df_pred[col] = df_pred[col].astype(np.float64)
    # Convert ageing time to ln(sqrt(x^2 + 1)) for prediction dataset
    df_pred['Transformed Ageing Time'] = [log(sqrt(i**2+1)) for i in list(df_pred['Transformed Ageing Time'])]
    
    for mech_prop in mech_props:
        # Create temporary drop list without one of properties + low VI col
        temp_drop_list = drop_list[:] + list(common_low_VIs)
        temp_drop_list.remove(mech_prop)
        # Drop useless features from the training dataset and N/A entry
        df = dataset.drop(temp_drop_list, axis=1).dropna()
        # Define the label set
        y = df[mech_prop]
        # Drop mech prop column from training dataset
        df = df.drop([mech_prop], axis=1)
        # One-hot encoding whole dataset (training + prediction)
        X_whole = pd.concat([df, df_pred], axis=0)
        X_whole = pd.get_dummies(X_whole)
        # Check whether prediction DF bring new categorical feature via encoded DF column size
        if X_whole.shape[1] != (pd.get_dummies(df).shape[1]):
            raise ValueError('Prediction dataset brings new categorical features, \
                              the size of one-hot encoded dataset does not match')
        # Split the original and prediction dataset
        X = pd.DataFrame(X_whole.head(len(df)))
        X_pred = pd.DataFrame(X_whole.tail(len(df_pred)))
        # Normalize the original dataset and apply to prediction set
        num_loc = 0
        for i in df.dtypes:
            if i == 'object': break
            else: num_loc += 1
        scale_features = X_whole.columns[0 : num_loc]
        scaler = preprocessing.StandardScaler().fit(X[scale_features])
        X[scale_features] = scaler.transform(X[scale_features])
        X_pred[scale_features] = scaler.transform(X_pred[scale_features])
        # Create target property dataset and store X and y in dictionary  
        dataset_dic[mech_prop] = [X, y]
        pred_dic [mech_prop] = X_pred
        
    return dataset_dic, pred_dic

def cal_fitness(known_data, population, para):
    '''
    known_data in DataFrame form
    population in np.array form
    '''
    # Define fitness list
    fitness_list = []
    
    # Calculate the cosine similarity between population and high-strength AA (YTS > 500) in dataset
    comp = ['Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr', 'Zn', 'Ti', 'Zr', 'Li', 'Sc']
    high_AA = known_data.loc[known_data['YTS'] > 500] # & (known_data['UTS'] > 600)
    high_AA_comp = np.array(high_AA[comp]) # Get composition of high-strength AA in dataset
    high_AA_comp = np.unique(high_AA_comp, axis=0) # Remove duplicated composition
    high_AA_comp_trans = np.transpose(high_AA_comp) # Transpose the high AA comp for further manipulation
    pop_comp = np.array([i[0:len(comp)] for i in population], dtype = np.float64) # Get composition of population
    
    dot_product = np.dot(pop_comp, high_AA_comp_trans) # Get dot product between pop and high-AA
    norm_AA = np.linalg.norm(high_AA_comp_trans, axis = 0) # Get norm of high AA
    # Expand norm of high AA into dot_product shape with same value in each col
    norm_AA = np.tile(norm_AA, (len(population), 1)) 
    norm_pop = np.linalg.norm(pop_comp, axis = 1) # Get norm of population
    norm_pop[norm_pop == 0] = 0.00001 # avoid the 0 as the denominator in cos_sim calculation
    # Expand norm of pop into dot_product shape with same value in each row
    norm_pop = np.transpose(np.tile(norm_pop, (len(high_AA_comp), 1)))
    
    cos_sim = dot_product / (norm_AA * norm_pop)
    mean_cos_sim = np.sum(cos_sim, axis = 1) / len(high_AA_comp)

    # Process training and prediction dataset for prediction
    Training_data, pred_data = Pred_DataProcessing(known_data, population)
    # Make prediction for all mechanical properties
    for prop in ['YTS','UTS','EL']:
        X = Training_data[prop][0]
        y = Training_data[prop][1]
        X_pred = pred_data[prop]
        if prop == 'YTS' or prop == 'UTS':
            model = svm.SVR(kernel='rbf', C = para[prop][0], gamma = para[prop][1])
        elif prop == 'EL':
            model = xgb.XGBRegressor(n_estimators = para[prop][0], max_depth = para[prop][1], 
                                     learning_rate = para[prop][2])
        # Make prediction and append to the fitness list
        model.fit(X, y)
        prediction = model.predict(X_pred)
        fitness_list.append(prediction)
    fitness_list.append(-mean_cos_sim)
    
    return fitness_list