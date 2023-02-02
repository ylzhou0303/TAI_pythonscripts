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
Depths = np.array([0.25, 0.2, 0.1, 0.075, 0.05, 0.015, 0.005, 0.003, 0.0020])  #depths of each layer
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


#%% plot the theoretical profile (no outgassing) vs the actual profile
# for day 29, the mean profile of theoretical increase in CH4 is
t = 29
MetThry_inc = np.mean(MetThry, axis = 0)[29,:] / (Depths * 0.0001 * Porosity * 1e3)   #calculate the mean profile and convert to mol/L

#calculate the mean actual profile
var_id = 2
MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Full_Data[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs)


MetThry_prof = MeanProfs[:,29] + MetThry_inc   #the theoretical profile of day30 is the actual profile on day29 plus the increase in CH4 per day


plt.plot(MetThry_prof, depths, 'k--')
plt.plot(MeanProfs[:,29], depths, 'r-')
plt.plot(MeanProfs[:,30], depths, 'k-')


#%% A second way to calculate the methane outflux
# I used a Tracer2 in PFLOTRAN to trace the balance between CH4 productin and oxidation
# So the concentration change of Tracer2 within 1 day is how much of CH4 changes if there is no outgassing
# The difference between that and the actual change of CH4 within 1 day is the outflux

# find out which column in the mass balance file is Tracer2
Tracer2_global_i = 9
Tracer2_bc_i = 26
Met_global_i = 10

#increase in Tracer2 in the modeled soil column
Tracer2_inc = (Mass[29,Tracer2_global_i] - Mass[28,Tracer2_global_i]) - Mass[28,Tracer2_bc_i]
#because the boundary condition conc. for Tracer2 had to be set up, some Tracer2 was taken out through the sed_air_interface, we need to add that back

#the actual methane change calculated by PFLOTRAN
Met_inc = Mass[29,Met_global_i] - Mass[28,Met_global_i]

#the difference would be the outflux of CH4
MetLoss_rate2 = (Tracer2_inc - Met_inc) / 0.01   #mol/m2/day

#%% relationship between O2 and CH4 flux
t = 29
X = Full_Data[300:400,t,1] / 2.5e-4 *100
Y = MetFlux2[:,t]

plt.rcParams.update({'font.size': 14})
plt.plot(X,Y, 'ro')
#plt.plot([0,50], [0.016,0], 'b-')
#joey = np.arange(0,50,1)
#yung = 0.00000735 * joey * joey - 0.00066 * joey + 0.0158
#plt.plot(joey,yung,'b--')
plt.xlabel('O2 %sat')
plt.ylabel('CH4 flux(mol m-2 d-1)')

#%% bar chart
t = 29
plt.bar(np.arange(1,50),MetFlux2[:,t], color = 'maroon', width = 0.7)
plt.xlabel('Column ID')
plt.ylabel('CH4 flux (mol m-2 d-1)')