# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:13:18 2023

@author: hughh
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Define the path of dataset
path = 'dataset.xlsx'

# Import initial data set and load compositions
df = pd.read_excel(path)
elem = ['Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr', 'Zn', 'Ti', 'Zr', 'Li', 'Sc']
comp = df[elem]

# Get index of data points with YTS > 500 MPa


# Import designed compositions in Loop I & II
L1_df = pd.read_excel('self-designed Al alloys.xlsx', sheet_name='Loop I')
L1_alloys = L1_df.iloc[[0, 2]]
L1_alloys_comp = L1_alloys[elem]

L2_df = pd.read_excel('self-designed Al alloys.xlsx', sheet_name='Loop II')
L2_alloys = L2_df.iloc[[0, 2]]
L2_alloys_comp = L2_alloys[elem]



'''PCA'''
# PCA composition of initial data points
pca = PCA(n_components=2)
comp_r = pca.fit(comp).transform(comp)

low_s_comp_r = comp_r[df['YTS'] < 500]
high_s_comp_r = comp_r[df['YTS'] >= 500]

# PCA for failed and successful alloys
L1_alloys_comp_r = pca.transform(L1_alloys_comp)
L1A1 = L1_alloys_comp_r[0]
L1A2 = L1_alloys_comp_r[1]
L2_alloys_comp_r = pca.transform(L2_alloys_comp)
L2A1 = L2_alloys_comp_r[0]
L2A2 = L2_alloys_comp_r[1]



# Plot the comparison
plt.figure(figsize=(8,8))

# plt.scatter(low_s_comp_r[:,0], low_s_comp_r[:,1], s=20, c='#154c79', alpha=0.8, 
#             label='Low-strength Data Points')

# plt.scatter(high_s_comp_r[:,0], high_s_comp_r[:,1], s=20, c='#De6d00', alpha=0.8, 
#             label='High-strength Data Points')

plt.scatter(comp_r[:,0], comp_r[:,1], s=40, c='#154c79', alpha=0.4, label='Initial Data Points')
plt.scatter(L1A1[0], L1A1[1], s=70, facecolors='none', edgecolors='#De6d00', 
            linewidth=3, alpha=0.8, marker='^', label='S1-1 D')
plt.scatter(L1A2[0], L1A2[1], s=70, facecolors='none', edgecolors='#De6d00', 
            linewidth=3, alpha=0.8, marker='v', label='S1-2 D')

plt.scatter(L2A1[0], L2A1[1], s=110, facecolors='none', edgecolors='#940709', 
            linewidth=3, alpha=0.8, marker='p', label='S2-1 D')
plt.scatter(L2A2[0], L2A2[1], s=130, facecolors='none', edgecolors='#940709', 
            linewidth=2, alpha=0.8, marker='*', label='S2-2 D')

plt.xlabel('PC1', fontsize=14, fontname='Arial', fontweight='bold')
plt.ylabel('PC2', fontsize=14, fontname='Arial', fontweight='bold')



plt.show()