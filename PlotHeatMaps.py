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


var_id = 4
var_str = Var_str[var_id]
layer = 6
i_start = (nz - layer) * nx * ny
i_end = i_start + nx * ny

if var_id == 1:
    conv = 1/2.5e-4*100
else:
    conv = 1e6

M = Full_Data[i_start:i_end, 30, var_id]  * conv  #extract the data to be inevestigated, by specifying the layer, timepoint, and variable id
A = M.reshape(nx, ny)      #this is a 10*10matrix, representing the view from top of the soil grids
B = np.flipud(A)   #flip upside down the matrix so that the grids with smaller y coordinates are at the bottom
                   #same as in the field

plt.rcParams.update({'font.size': 12})
plt.imshow(B, cmap ="plasma")
plt.xticks(ticks = np.arange(0,10), labels = np.arange(1,11))
plt.yticks(ticks = np.arange(0,10), labels = np.arange(1,11))
plt.xlabel('X coordinate (cm)')
plt.ylabel('Y coordinate (cm)')
cbar = plt.colorbar()
#cbar.ax.get_yaxis().set_ticks(np.arange(850,1650,200))
#cbar.set_label('CH4(μM)'.translate(subscript))


subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
superscript = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
#plt.title(var_str[6:len(var_str)-4] + ' (μM)')
#plt.title('O2 saturation (%)')
plt.title('SO4'.translate(subscript) + '2-(μM)'.translate(superscript))
#plt.title('CH4(μM)'.translate(subscript))
#plt.title('DOC (μM)')
#plt.title('H2S(aq) (μM)'.translate(subscript))

# set more parameters for the axis
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off


#%% compare the concentration between different grids
plt.bar(list(range(0, nx*ny)), M, color = 'skyblue', width = 0.4)
plt.title(Var_str[var])
plt.ylabel(Var_str[var])
plt.xlabel('Grid IDs')


