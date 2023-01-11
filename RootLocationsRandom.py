# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:19:14 2022

@author: YZ60069
"""

#%% Generate radomized numbers for the location of roots
import random
import numpy as np
nx = 7
ny = 7

#Define matrices of X and Y coordinates
XCoords = np.empty (shape = (nx,ny))
for i in range(0,nx):
    XCoords[i, :] = np.arange(0.07, 0.98, 0.14)


YCoords = np.empty (shape = (nx,ny))
for i in range(0,ny):
    YCoords[:, i] = np.arange(0.07, 0.98, 0.14)


# Generate random numbers for the ID of the grids containing root
RootGrids = np.empty( shape = (1,9) , dtype = np.int32)
RootLocs = np.empty( shape = (2,9) )

for i in range(0,9):
    RootGrids[0, i] = random.randint(1, 49)   # generate the ID of the subgrids
    idx_line = (RootGrids[0, i] - 1) // nx   # locate which x and y coordinates to take
    idx_row = (RootGrids[0, i] - 1) % nx
    
    RootLocs[0,i] = XCoords[idx_line, idx_row] #the X coordinate of the root location
    RootLocs[1,i] = YCoords[idx_line, idx_row] #the y coordinate of the root location
