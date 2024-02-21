# mitigating_data_bias_for_Al_alloys_design
Data repository and code for the paper "Designing unique and high-performance Al alloys via machine learning: mitigating data bias through active learning"

dataset.xlsx: initial dataset containing all Al alloys
self-designed Al alloys.xlsx: newly designed Al alloys during design Loop I & II
3D Plot.py: plot the initial dataset and newly designed Al alloys in a 3D map concerning their mechanical properties
Distribution_of_Data_Points_2D.py: use PCA to convert the composition of the initial dataset and newly designed Al alloys into a 2D plot
DataPreprocessing.py: preprocessing the dataset for ML modelling
FeatureEngineering.py: eliminate trivial features in the dataset
TrainingFunction.py: functions defined to train different ML models
ModelSelection.py: select the ML model with the best generalizability
ModelTraining.py: train the best model with the entire dataset based on the cross-validation method and determine the optimal hyperparameters
MOGA_algo.py: algorithm used for MOGA implementation, including the function calculating the fitness of the population
pymoo_NSGAII.py: MOGA based on NSGAII algorithm from pymoo package
