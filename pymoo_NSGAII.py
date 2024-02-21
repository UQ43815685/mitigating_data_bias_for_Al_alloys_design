# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 11:33:01 2022

@author: Hugh
"""

import os
# import time

import numpy as np
import pandas as pd

from math import log, sqrt
from MOGA_algo import cal_fitness
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.problem import Problem



'''Define the dataset and hyperparameters'''
# Define database  path
path = 'dataset.xlsx'

# Read and process database in DataFrame - dataset dictionary = {property:[feature, target property],...}
data = pd.read_excel(path)

# Common low VI features & Useless column in dataset
low_VI_list = ['Ga', 'Bi', 'B', 'Pb', 'Ni', 'Be', 'V']
drop_list = ['Alloy','YTS','UTS','EL','Al (min.)','Reference','Designation']

# Define the list of feature name
feature_list = data.drop(drop_list + low_VI_list, axis=1).columns.tolist()

# Convert ageing time to ln(sqrt(x^2 + 1))
data['Ageing Time'] = [log(sqrt(i**2+1)) for i in list(data['Ageing Time'])]
data = data.rename(columns = {'Ageing Time': 'Transformed Ageing Time'})

# Optimal model parameters for different dataset
para = {'YTS': [618, 0.08], 'UTS': [481, 0.06], 'EL': [1000, 4, 0.1]}

# Define the objective list
mech_props = ['YTS', 'UTS','EL']



'''Define the GA search space'''
# Search space in dictionary with - feature: [option list]
search_space = {'Si': np.arange(0, 12, 0.5), 'Fe': np.arange(0, 1.7, 0.1),
                'Cu': np.arange(0, 6.6, 0.2), 'Mn': np.arange(0, 2.2, 0.1),
                'Mg': np.arange(0, 6.0, 0.2), 'Cr': np.arange(0, 0.55, 0.05),
                'Zn': np.arange(0, 9.0, 0.2), 'V': np.arange(0, 0.25, 0.05),
                'Ti': np.arange(0, 0.21, 0.01), 'Zr': np.arange(0, 0.45, 0.05),
                'Li': np.arange(0, 4.0, 0.2), 'Ni': np.arange(0, 2.2, 0.2),
                'Ga': np.arange(0, 0.035, 0.005), 'Bi': np.arange(0, 0.7, 0.1),
                'Pb': np.arange(0, 0.7, 0.1), 'B': np.arange(0, 0.08, 0.02),
                'Be': np.arange(0, 0.15, 0.05), 'Sc': np.arange(0, 1.6, 0.1),
                'S/A Temp': np.arange(430.0, 530.0 ,10.0), 'Ageing Temp':np.arange(120.0, 180.0, 10.0), 
                'Ageing Time': np.arange(0, 105.0, 2.0)}

# feature range in dictionary with - feature: [[lower bound, upper bound], step]
variable_range = {'Si': [[0, 30], 0.4], 'Fe': [[0, 16], 0.1], 'Cu': [[0, 31], 0.2], 
                  'Mn': [[0, 20], 0.1], 'Mg': [[0, 30], 0.2], 'Cr': [[0, 10], 0.05],
                  'Zn': [[0, 43], 0.2], 'V': [[0, 5], 0.05], 'Ti': [[0, 4], 0.05],
                  'Zr': [[0, 8], 0.05], 'Li': [[0, 0], 0.2], 
                  'Ni': [[0, 11], 0.2], 'Ga': [[0, 8], 0.005], 'Bi': [[0, 7], 0.1], 
                  'Pb': [[0, 7], 0.1], 'B': [[0, 4], 0.02], 'Be': [[0, 3], 0.05], 
                  'Sc': [[0, 10], 0.05], 'S/A Temp': [[46, 53], 10.0], 
                  'Ageing Temp':[[11, 15], 10.0], 'Ageing Time': [[0, 24], 4.0]}

# Remove low VI feature from search space and variable range dictionary
for i in low_VI_list:
    search_space.pop(i, None)
    variable_range.pop(i, None)
    
# Define the lower bound, upper bound, and step size of each feature into lists
lower_bound = [variable_range[i][0][0] for i in variable_range]
upper_bound = [variable_range[i][0][1] for i in variable_range]
step_size = [variable_range[i][1] for i in variable_range]

# Define the processing features
process_para = np.array(['Extruded', 'Water', 'No', 'Artificial', 'No', 'Rod'], dtype = object)

# Transform variable from integer to physical value & append process parameters
def int2real(int_list):
    transformed_list = []
    for i in int_list:
        transformed = np.hstack((np.round_(i * step_size, 4), process_para))
        transformed_list.append(transformed)
    return transformed_list



'''Define the fitness function'''
def fitness(data_dic, population, para):
    # Calculate different fitness types of population
    fitness_list = cal_fitness(data_dic, population, para)
        
    # Group different fitness values related to same instance into list
    scores_list = []
    for i in range(len(population)):
        scores = []
        for k in range(len(para) + 1):
            scores.append(fitness_list[k][i])
        scores_list.append(scores)
    scores_list = np.array(scores_list)
    
    return np.negative(scores_list)



'''Define the Problem class'''
class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var = len(search_space),
                         n_obj = len(mech_props) + 1, # Plus cosine similarity as new obj
                         xl = lower_bound,
                         xu = upper_bound,
                         type_var = int)
    
    def _evaluate(self, x, out, *args, **kwargs):
        transformed_x = int2real(x)
        # start = time.process_time()
        scores = fitness(data, transformed_x, para)
        # print(time.process_time() - start)
        out['F'] = np.array(scores)

# Define the GA problem, algorithm, stop criteria
problem = MyProblem()
pop_size = 2000
algorithm = NSGA2(pop_size = pop_size,
                  sampling = IntegerRandomSampling(),
                  crossover = SBX(prob = 0.8, eta = 0.5, vtype = float, repair = RoundingRepair()),
                  mutation = PM(eta = 0.5, vtype = float, repair = RoundingRepair()),
                  eliminate_duplicates = True)
stop_criteria = ('n_gen', 500)

# Start GA searching
results = minimize(problem = problem, algorithm = algorithm, termination = stop_criteria, verbose=True)