# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 18:22:06 2022

@author: YZ60069
"""

#%% This file analyzes the biogeochemistry of each grid under O2 spatial heterogeneity
# to investigate how O2 heterogeneity is influencing the biogeochemistry

import pandas as pd


#%% Heat map for different variables concentrations at different depths
nx = 10
ny = 10
nz = 9


var = 5
var_str = Var_str[var]
layer = 6
i_start = (nz - layer) * nx * ny
i_end = i_start + nx * ny

M = Full_Data[i_start:i_end, 30, var]  / 2.5e-4 * 100  #extract the data to be inevestigated, by specifying the layer, timepoint, and variable id
A = M.reshape(nx, ny)      #this is a 7*7matrix, representing the view from top of the soil grids
B = np.flipud(A)   #flip upside down the matrix so that the grids with smaller y coordinates are at the bottom
                   #same as in the field
plt.imshow(B, cmap ="Reds")
ax = plt.gca()
ax.grid(color='red', linestyle='-.', linewidth=0)
plt.colorbar()
plt.title("O2 saturation (%)")
plt.title(var_str[0:len(var_str) - 4] + ' uM')

#%% compare the concentration between different grids
plt.bar(list(range(0, nx*ny)), M, color = 'skyblue', width = 0.4)
plt.title(Var_str[var])
plt.ylabel(Var_str[var])
plt.xlabel('Grid IDs')


