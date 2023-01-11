# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:19:14 2022

@author: YZ60069
"""

#%% Generate radomized numbers for the location of roots
import random
import numpy as np
nx = 10
ny = 10
nroots = 20

#Define matrices of X and Y coordinates
XCoords = np.zeros (shape = (nx,ny))
for i in range(0,nx):
    XCoords[i, :] = np.arange(0.005, 0.096, 0.01)


YCoords = np.zeros (shape = (nx,ny))
for i in range(0,ny):
    YCoords[:, i] = np.arange(0.005, 0.096, 0.01)


#%% Generate random numbers for the ID of the grids containing root
RootGrids = np.zeros(shape = (nroots), dtype = np.int0)
i = 0

while 0 in RootGrids:
    r = random.randint(0, nx*ny -1 )
    if r not in RootGrids:
        RootGrids[i] = r
        i = i + 1

#%% find out the coordinates of these grids with roots  
X = XCoords.reshape((nx*ny))
Y = YCoords.reshape((nx*ny))

RootLocs = []
for i in range(nroots):
    RootLocs.append([X[RootGrids[i]], Y[RootGrids[i]]])

RootLocs = np.around(np.array(RootLocs),3)



#%% compile for the PFLOTRAN input
Strs = ''
depth = 0.6

for i in range(nroots):
    
    temp_str = ('REGION root' + str(i + 1) + '\n\tCOORDINATES\n\t\t' + str(RootLocs[i,0]) + '  ' + str(RootLocs[i,1])
                + '  ' + str(depth) + '\n\t\t' + str(RootLocs[i,0]) + '  ' + str(RootLocs[i,1]) + '  ' + str(depth)
                + '\n\t/\n\tFACE TOP\nEND')
                
    Strs = Strs + '\n\n' + temp_str

