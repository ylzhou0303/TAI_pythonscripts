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


var_id = 1
var_str = Var_str[var_id]
layer = 6

i_start = (nz - layer) * nx * ny
i_end = i_start + nx * ny

if var_id == 1:
    conv = 1/2.5e-4*100
else:
    conv = 1e3

M = Full_Data[i_start:i_end, 10, var_id]  * conv  #extract the data to be inevestigated, by specifying the layer, timepoint, and variable id
A = M.reshape(nx, ny)      #this is a 10*10matrix, representing the view from top of the soil grids
B = np.flipud(A)   #flip upside down the matrix so that the grids with smaller y coordinates are at the bottom
                   #same as in the field


plt.rcParams.update({'font.size': 14})

if var_id == 1:
    plt.imshow(B, cmap = 'plasma')#, vmin = 0, vmax = 100)       #O2
elif var_id == 2:
    plt.imshow(B, cmap = 'plasma', vmin = 0.5, vmax = 1.25)    #CH4
elif var_id == 3:
    plt.imshow(B, cmap = 'plasma', vmin = 3.5, vmax = 8)     #DOC
elif var_id == 4:
    plt.imshow(B, cmap = 'plasma', vmin = 1, vmax = 7)     #SO42-
elif var_id == 5:
    plt.imshow(B, cmap = 'plasma', vmin = 0.5, vmax = 6)     #H2S



#plt.imshow(B, cmap = "plasma")

if var_id == 1:
    
    threshold = 0.1
    above_threshold_mask = B >= threshold
    
    num_rows, num_cols = B.shape
    for i in range(num_rows):
        for j in range(num_cols):
            if above_threshold_mask[i, j]:
                # Highlight the edge of the grid cell above the threshold
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='white', linewidth=1)
                plt.gca().add_patch(rect)


plt.xticks(ticks = np.arange(-0.5,10.5), labels = np.arange(0,11))
plt.yticks(ticks = np.arange(-0.5,10.5), labels = np.arange(10,-1,-1))
plt.xlabel('X coordinate (cm)')
plt.ylabel('Y coordinate (cm)')
cbar = plt.colorbar()
#cbar.ax.get_yaxis().set_ticks(np.arange(850,1650,200))
#cbar.set_label('CH4(μM)'.translate(subscript))

sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
sup = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

if var_id == 1:
    titletxt = 'O2'.translate(sub)
    cbar.set_label('O2 saturation (%)'.translate(sub))
elif var_id == 2:
    titletxt = 'CH4'.translate(sub)
    cbar.set_label('CH4 (mmol'.translate(sub) + ' L-1)'.translate(sup))
elif var_id == 3:
    titletxt = 'DOC'
    cbar.set_label('DOC (mmol L-1)'.translate(sup))
elif var_id == 4:
    titletxt = 'SO4'.translate(sub) + '2-'.translate(sup)
    cbar.set_label( 'SO4'.translate(sub) + '2-(mmol L-1)'.translate(sup))
elif var_id == 5:
    titletxt= 'H2S'.translate(sub)
    cbar.set_label('H2S(aq) (mmol'.translate(sub) + ' L-1)'.translate(sup))
elif var_id == 6:
    titletxt = 'Tracer2 (μM)'


plt.title( '(b' + str(var_id) + ') ' + titletxt)




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


#%% plot to show the location of O2 injection points
nx = 10
ny = 10
nz = 9


var_id = 1
var_str = Var_str[var_id]
layer = 6

i_start = (nz - layer) * nx * ny
i_end = i_start + nx * ny

if var_id == 1:
    conv = 1/2.5e-4*100
else:
    conv = 1e3

M = Full_Data[i_start:i_end, 10, var_id]  * conv  #extract the data to be inevestigated, by specifying the layer, timepoint, and variable id
A = M.reshape(nx, ny)      #this is a 10*10matrix, representing the view from top of the soil grids
B = np.flipud(A)   #flip upside down the matrix so that the grids with smaller y coordinates are at the bottom
                   #same as in the field

threshold = 1000
below_threshold = B < threshold
above_threshold = B >= threshold

cmap= plt.cm.colors.ListedColormap(['#F8E0F1','#86C440'])


plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 18})
plt.imshow(below_threshold, cmap = cmap, alpha = 1)
plt.imshow(above_threshold, cmap = cmap, alpha = 1)


num_rows, num_cols = B.shape
for i in range(num_rows - 1):
    plt.axhline(i + 0.5, color='black', linewidth=1)
for j in range(num_cols - 1):
    plt.axvline(j + 0.5, color='black', linewidth=1)


plt.xticks(ticks = np.arange(-0.5,10.5), labels = np.arange(0,11))
plt.yticks(ticks = np.arange(-0.5,10.5), labels = np.arange(10,-1,-1))
plt.xlabel('X coordinate (cm)')
plt.ylabel('Y coordinate (cm)')
#cbar = plt.colorbar()
#cbar.ax.get_yaxis().set_ticks(np.arange(850,1650,200))
#cbar.set_label('CH4(μM)'.translate(subscript))

