import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



'''Plot samples in data set'''
# Define database path
path = 'dataset.xlsx'

# Load DataFrame and Properties
df = pd.read_excel(path)
YTS = df['YTS'].tolist()
UTS = df['UTS'].tolist()
EL = df['EL'].tolist()

# Find PF of no-nan properties dataset
props = np.array([YTS, UTS, EL]).transpose()
nonan = ~np.isnan(props).any(axis=1)

props = props[nonan]
props = props.transpose() # Transpose array for drawing

# Load GA designed alloys during design Loop I & II
L1_df = pd.read_excel('self-designed Al alloys.xlsx', sheet_name='Loop I')
L1A1 = L1_df.iloc[[0]]
L1A2 = L1_df.iloc[[2]]

L2_df = pd.read_excel('self-designed Al alloys.xlsx', sheet_name='Loop II')
L2A1 = L2_df.iloc[[0]]
L2A2 = L2_df.iloc[[2]]



# Draw the dataset in 3D
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')

# All data points in data set
ax.scatter(props[0], props[1], props[2], s=30, c='#154c79', alpha=0.4, label='Initial Data Points')

# GA failed designed alloys with prediction result
ax.scatter(L1A1['YTS'], L1A1['UTS'], L1A1['EL'], s=80, facecolors='none', edgecolors='#De6d00', 
           linewidth=3, alpha=0.8, marker='^', label='L1A1 D')
ax.scatter(L1A2['YTS'], L1A2['UTS'], L1A2['EL'], s=80, facecolors='none', edgecolors='#De6d00', 
           linewidth=3, alpha=0.8, marker='v', label='L1A2 D')

# GA successfully designed alloys with prediction
ax.scatter(L2A1['YTS'], L2A1['UTS'], L2A1['EL'], s=110, facecolors='none', edgecolors='#940709', 
           linewidth=3, alpha=0.8, marker='p', label='L2A1 D')
ax.scatter(L2A2['YTS'], L2A2['UTS'], L2A2['EL'], s=130, facecolors='none', edgecolors='#940709', 
           linewidth=2, alpha=0.8, marker='*', label='L2A2 D')

# Set axis label for x & y and linewidth
ax.set_xlabel('YTS (MPa)', fontsize=14, fontname='Arial', fontweight='bold', labelpad=10)
ax.set_ylabel('UTS (MPa)', fontsize=14, fontname='Arial', fontweight='bold', labelpad=10)
ax.set_zlabel('EL(%)', fontsize=14, fontname='Arial', fontweight='bold')
for axis in [ax.xaxis, ax.yaxis]:
    axis.line.set_linewidth(2)

# Set axis tick and make zaxis invisible
ax.tick_params(which='major', width=2, length=6, labelsize=12)
ax.axes.set_xlim3d(left=0, right=700) 
ax.axes.set_ylim3d(bottom=0, top=750) 
ax.axes.set_zlim3d(bottom=0, top=50) 

ax.set_yticklabels([0, 100, 200, 300, 400, 500, 600, 700],
                    verticalalignment='baseline',
                    horizontalalignment='left')

# Set grid pattern
ax.xaxis._axinfo['grid']['linestyle'] = "--"
ax.yaxis._axinfo['grid']['linestyle'] = "--"
ax.zaxis._axinfo['grid']['linestyle'] = "--"

# Draw cut-through surface to view data points with YTS > 500 MPa
x1 = 500
y1 = np.arange(500, 750, 1)
z1 = np.arange(0, 20, 0.5)
Y1, Z1 = np.meshgrid(y1, z1)
X1 = np.full_like(Y1, x1)

ax.plot_surface(X1, Y1, Z1, alpha=0.2, linewidth=0, color='#De6d00')


ax.view_init(elev = 10, azim = -70)