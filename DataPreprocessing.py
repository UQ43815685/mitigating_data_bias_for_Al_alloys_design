import pandas as pd
from sklearn import preprocessing
from math import log, sqrt



def AgeingTimeTrans(dataset):
    '''Convert ageing time to ln(sqrt(x^2 + 1))'''
    Ageing_Time = list(dataset['Ageing Time'])
    transformed = [log(sqrt(i**2+1)) for i in Ageing_Time]
    dataset['Ageing Time'] = transformed
    dataset = dataset.rename(columns = {'Ageing Time': 'Transformed Ageing Time'})
    return dataset

def RF_DataPreprocessing(dataset):
    """
    Preprocess dataset for feature engineering step with random forest
    Return:
        dataset dictionary = {property:[feature, target property],...}
    """
    
    # Convert ageing time to ln(sqrt(x^2 + 1))
    dataset = AgeingTimeTrans(dataset)
    
    # Preprocess dataset
    mech_props = ['YTS','UTS','EL']
    drop_list = ['Alloy','YTS','UTS','EL','Al (min.)','Reference','Designation']
    dataset_dic = {}
    for mech_prop in mech_props:
        # Create temporary drop list without one of mechanical properties
        temp_drop_list = drop_list[:]
        temp_drop_list.remove(mech_prop)     
        # Drop useless columns and rows in the dataset
        df = dataset.drop(temp_drop_list, axis=1)
        df = df.dropna() # drop row without property value
        # Create one-hot encoded feature dataset
        X = df.drop(mech_prop, axis=1)
        X = pd.get_dummies(X)
        # Normalize the feature dataset by z-score
        num_loc = 0
        for i in df.dtypes:
            if i == 'object': break
            else: num_loc += 1
        scaler = preprocessing.StandardScaler()
        scale_features = X.columns[0:num_loc]
        X[scale_features] = scaler.fit_transform(X[scale_features])
        # Create target property dataset and store X and y in dictionary
        y = df[mech_prop]    
        dataset_dic[mech_prop]=[X, y]
        
    return dataset_dic
    
def SVR_DataPreprocessing(preprocessed_dataset, common_low_VIs):
    '''
    Preprocess dataset for ML modelling step with SVR
    Input (in previous step):
        preprocessed dataset dictionary = {property:[feature, target property],...}
        common_low_VIs = [features with low VI in all three properties]
    Return:
        dataset dictionary = {property:[feature, target property],...}
    '''
    for mech_prop in preprocessed_dataset:
        # Load feature and targeted property dataset to variables
        dataset = preprocessed_dataset[mech_prop]
        X = dataset[0]
        y = dataset[1]
        # Drop commom low-VI features in feature dataset
        X = X.drop(common_low_VIs, axis=1)
        # Replace feature dataset with dropped feature dataset
        preprocessed_dataset[mech_prop] = [X, y]  
    return preprocessed_dataset

def Pred_DataProcessing(dataset, dataset_pred, common_low_VIs, ATT = False):
    """
    Preprocess training and prediction dataset 
    Return:
        dataset dictionary = {property:[feature, target property],...}
        prediction dictionary = {property: feature}
    """
    
    # Convert ageing time to ln(sqrt(x^2 + 1)) if True
    if ATT:
        # For training dataset & prediction dataset
        dataset = AgeingTimeTrans(dataset)
        dataset_pred = AgeingTimeTrans(dataset_pred)

    # Preprocess training and prediction dataset
    mech_props = ['YTS','UTS','EL']
    drop_list = ['Alloy','YTS','UTS','EL','Al (min.)','Reference','Designation']
    dataset_dic = {}
    pred_dic = {}
    
    for mech_prop in mech_props:
        # Create temporary drop list without one of properties + low VI col
        temp_drop_list = drop_list[:] + list(common_low_VIs)
        temp_drop_list.remove(mech_prop)
        # Drop useless features from the original dataset and drop entries with N/A
        df = dataset.drop(temp_drop_list, axis=1).dropna()
        # Define the label set
        y = df[mech_prop]
        # Drop useless features from prediction dataset 
        df_pred = dataset_pred.drop(temp_drop_list, axis=1)
        # One-hot encoding whole dataset (training + prediction)
        X_whole = pd.concat([df, df_pred], axis=0)
        X_whole.drop(mech_prop, axis=1, inplace=True)
        X_whole = pd.get_dummies(X_whole)
        # Check whether prediction DF bring new categorical feature via encoded DF column size
        if X_whole.shape[1] != (pd.get_dummies(df).shape[1] - 1):
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