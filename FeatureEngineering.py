from sklearn.ensemble import RandomForestRegressor

def RF_FeatureEngineering(preprocessed_data):
    """
    Engineer low VI features with random forest
    Return:
        list: common low VI features
    """
    mech_props = ['YTS','UTS','EL']
    one_hot_feature = ['ManufacturingMethod', 'HeatTreatmentMedium',
                       'StrainHardeningMethod', 'AgeingType',
                       'TreatmentAfterward', 'ProductionShape']
    low_VI_lists = []
    
    for mech_prop in mech_props:
        # Load dataset of feature and target property & get feature column names
        X = preprocessed_data[mech_prop][0]
        y = preprocessed_data[mech_prop][1]
        EncodedFeature = list(X.columns)

        # Construct and train random forest model and get VIs
        clf = RandomForestRegressor(n_estimators = 1000, criterion='absolute_error', n_jobs=-1)
        clf.fit(X, y)
        gini_importance = clf.feature_importances_
        
        # Add gini importance of one-hot encoded feature together
        ungrouped_VI = []
        for VI in zip(EncodedFeature, gini_importance):
            ungrouped_VI.append(VI)
        
        grouped_VI = []
        for feature in one_hot_feature:
            total_VI = 0
            for VI in zip(EncodedFeature, gini_importance):
                if feature in VI[0]:
                    total_VI += VI[1]
                    ungrouped_VI.remove(VI)
            grouped_VI.append((feature, total_VI))    
        grouped_VI = ungrouped_VI + grouped_VI
        
        # Append grouped VI which lower than 1% into low VI list
        low_VI_list = []
        for VI in grouped_VI:
            if VI[1] < 0.01:
                low_VI_list.append(VI[0])
        
        low_VI_lists.append(low_VI_list)
    
    # Find common low VI features in all three properties
    common_low_VIs = set(low_VI_lists[0]) & set(low_VI_lists[1]) & set(low_VI_lists[2])
        
    return common_low_VIs