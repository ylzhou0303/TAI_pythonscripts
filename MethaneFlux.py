# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:08:31 2023

@author: YZ60069
"""

#%% This file calculates the flux of CH4 for each column
# First calculates how much of CH4 is supposed to be in the column if there is no outgassing 
# based on concentrations and reaction rates
# Next subtract the actual CH4 amount from that, the difference is the flux

#%% 
Depths = [0.25, 0.2, 0.1, 0.075, 0.05, 0.015, 0.005, 0.003, 0.0020]  #depths of each layer
Area = 0.01 * 0.01    #area of each grid
Porosity = 0.8

MetThry = np.zeros(shape = (nx*ny, ntimepoint, nz))
MetAmt = np.zeros(shape = (nx*ny, ntimepoint, nz))
for z in range(0,nz):  #calculate for all grids for each layer, one layer by one layer
    i_start = (nx * ny) * z
    i_end = (nx * ny) * (z + 1)
    Volume = Area * Depths[z]   #volume of each grid at this layer, unit:m3
    
    #calculate the amount of methane in theory if no outgassing (no ebullition and no diffusion out)
    MetGen = Rates[i_start:i_end,:,1] * 3600 * 24 * 1e3 * Volume * Porosity  #calculate the amount of methane produced , unit: mol (the unit of rate is mol L-1 s-1)
    MetOxi = Rates[i_start:i_end,:,2] *3600 * 24 * 1e3 * Volume  * Porosity
    MetThry[:,:,z] = MetGen - MetOxi      
    
    #calculate the actual total amount of methane within each grid, produced within one day
    MetAmt[:,:,z] = Full_Data[i_start:i_end,:,2]  * 1e3 * Volume * Porosity


MetThry_col = np.sum(MetThry, axis = 2)   #The amount methane within each column produced within one day, if there is no outgassing
MetAmt_col = np.sum(MetAmt, axis = 2)
MetActl_col = MetAmt_col[:,1:ntimepoint] - MetAmt_col[:,0:ntimepoint-1]   #the actual increase in methane within one day given by PFLOTRAN 



#%% Step 2: calculate the difference between the theoretical and actual amount of CH4 

#calculate the difference, which would be how much of CH4 was out
MetFlux = MetThry_col[:,0:ntimepoint-1] - MetActl_col   #unit: mol d-1

MetFlux2= MetFlux / Area   #unit: mol m-2 d-1


#%% plot the heatmaps
t = 29
M = MetFlux2[:, t]  #extract the data to be inevestigated, by specifying the timepoint
A = M.reshape(nx, ny)      #this is a 7*7matrix, representing the view from top of the soil grids
B = np.flipud(A)   #flip upside down the matrix so that the grids with smaller y coordinates are at the bottom
                   #same as in the field
plt.imshow(B, cmap ="Reds")
plt.colorbar()
plt.title('Methane Flux by column (mol m-2 day-1)')



#%% calculate the total flux
t = 29
MetThry_sum = sum(MetThry_col[:,t])    #total amount of CH4 produced within one day in theory if no outgassing, unit:mol
MetActl_sum = sum(MetActl_col[:,t])    #total amount of CH4 produced within one day calculated from PFLOTRAN

MetLoss = MetThry_sum - MetActl_sum   #total methane loss, including diffusion via sed_air-interface and ebullition
MetLoss_rate = MetLoss / 0.01          #methane loss rate, unit: mol/m2/day, 0.01 is the domain area




#%% relationship between O2 and CH4 flux
t = 29
X = Full_Data[i_start:i_end,t,1] / 2.5e-4 *100
Y = MetFlux2[:,t]
plt.plot(X,Y, 'ro')
#plt.plot([0,50], [0.016,0], 'b-')
joey = np.arange(0,50,1)
#yung = 0.00000735 * joey * joey - 0.00066 * joey + 0.0158
#plt.plot(joey,yung,'b--')
plt.xlabel('O2 %sat')
plt.ylabel('CH4 flux(mol m-2 d-1)')

#%% bar chart
t = 29
plt.bar(np.arange(1,50),MetFlux2[:,t], color = 'maroon', width = 0.7)
plt.xlabel('Column ID')
plt.ylabel('CH4 flux (mol m-2 d-1)')