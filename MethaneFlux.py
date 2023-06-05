# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:08:31 2023

@author: YZ60069
"""

#%% This file calculates the flux of the threshold-triggered removal of CH4 
# I set up an artificial reaction in PFLOTRAN to simulate the ebullition process, in which CH4 is removed when CH4 concentration is above the threshold
# I have three different methods to calculate the ebullition flux below

#%% Ebullition
# Method 1： Calculate the difference between theorectical CH4 concentration and the threshold

#Import parameters of the simulation domain
#Thickness = np.reshape([[0.25]*100, [0.2]*100, [0.1]*100, [0.075]*100, [0.05]*100, [0.015]*100, [0.005]*100, [0.003]*100, [0.002]*100] , (1,900))
Thickness = np.reshape([[0.25]*100, [0.2]*100, [0.1]*100, [0.03]*100, [0.03]*100,[0.03]*100, [0.02]*100,[0.015]*100, [0.015]*100, [0.005]*100, [0.003]*100, [0.002]*100] , (1,1200))
Thickness = np.transpose(Thickness)   #thickness of each grid cell
Area = 0.01 * 0.01    #area of each grid
Porosity = 0.8

# 1) Calculate the theoretical CH4 concentration assuming there is no ebullition
# Because the concentration of CH4 calculated by PFLOTRAN would always be lower than the threshold (otherwise the extra would be removed),
# I could not use the CH4 concentration reported by PFLOTRAN to calculate the rate of ebullition.
# I need to first calculate what the theoretical CH4 concentration on next day would be if there is no ebullition.

ch4_today = Full_Data[:, 29, 2]  #extract the CH4 concentration on day 29
ch4_increase = (Rates[:, 29, 1] - Rates[:, 29, 2]) * 3600 * 24   #the difference between methanogenesis rate and methane oxidation rate is the net increase in CH4 concentration
ch4_hypotmr = ch4_today + ch4_increase   #the hypothetical CH4 concentration on day30 if there is no ebullition going on  

# 2) Calculate the extra CH4 relative to the threshold concentration, which would be the ebullition efflux in my model framework
Cth = 0.0014236   #threshold concentration, unit: mol L-1
Diff = (ch4_hypotmr - Cth).reshape(1200,1)     #the difference between the hypothetical CH4 concentration and threshold
Trigger = (Diff > 0) * 1                      # judge if and where the hypothetical CH4 concentration is above threshold, if yes, assign 1, if not, assign 0

Volume = Thickness * Area * Porosity           #water volume of each of the 900 grid cells
Ebl_cell = Diff * 1e3 * Volume * Trigger    #convert to mol/m3, multiplied by the water volume of each cell to calculate the amount of CH4 removed by ebullition from each cell
                                                    #only do the calculation for the cells with a positive concentration difference, as CH4 is only removed when the CH4 concentration is above threshold, unit: mol d-1

Ebl_domain = np.sum(Ebl_cell)   # add the CH4 removal from each cell together, i.e., the CH4 ebullition flux from the simulation domain, unit: mol d-1
Ebl_areal = Ebl_domain * 1e3 / (0.1 * 0.1) #divided by the surface area of the simulation domain to convert to mmol m-2 d-1


#%% 3) Plant-mediated CH4 transport
ch4_conc = Full_Data[:, 29, 2]   #extract the CH4 conc
tracer2_conc = Full_Data[:, 29, 6]   #extract the tracer2 conc
Cell_area = 0.01 * 0.01    #surface area of each cell, unit: m2
Cell_vol = Cell_area * Thickness * 0.8    #water volume of each cell, 0.8 is the porosity

print('Have you specified the correct mode yet????\n'*3)

mode = 'Homo'

if mode == 'NO':
   PlantF_areal = 0
else:
    if mode == 'Homo':
        Cth_tracer2 = 1e-7             # the tracer2 threshold for plant-mediated CH4 transport for the Homo mode
        Vmax = 1.4e-8                  # maximum reaction rate for the reaction simulating homogeneous plant-mediated CH4 transport
        HSC = 1e-2
    elif mode == 'Het':                 
        Cth_tracer2 = 1e-4             # tracer2 threshold for the heterogeneous mode of root setup (1e-4 mol/L-1), use 1e-5 when doing sensitivity analysis with a high diffusion coefficient
        Vmax = 9.3e-8                  # maximum rate for heterogeneous mode
        HSC = 1e-2
    
    Trigger = (tracer2_conc > Cth_tracer2) * 1        #judge if the tracer2 concentration is above the threshold, if yes, assign 1, if not assign 0
    Monod = ch4_conc / (ch4_conc + HSC)
    PlantF_rate = Vmax * Monod * Trigger    #calculate the reaction rate for each root cell, the non-root cells would multiply 0, unit: mol L-1 s-1
    PlantF_cell = PlantF_rate * 1e3 * (3600*24) * Volume      # the removal of CH4 from each cell for one day, mol cell-1 d-1
    PlantF_domain = np.sum(PlantF_cell)               # the flux from the domain, mol d-1
    PlantF_areal = PlantF_domain * 1e3 / (0.1 * 0.1)  #divided by the surface area of the simulation domain, convert to mmol m-2 d-1

#%% Compile fluxes to calculate the total flux
import pandas as pd
MetDiffusion = - Mass[29,27]*1e3/0.01   # the diffusion flux via sediment-air interface, unit: mmol m-2 d-1
MetF_total = MetDiffusion + Ebl_areal + PlantF_areal

MetF = {'Total Flux': MetF_total, 'Diffusion': MetDiffusion, 'Plant-mediated': PlantF_areal, 'Ebullition': Ebl_areal}
MetF = pd.DataFrame(MetF, index = ['Flux'])
print(MetF)




#%%
plt.plot(ch4_today.reshape(9,100).mean(axis = 1) * 1e3, depths, 'b-', label = 'CH4 day 29')
plt.plot(ch4_hypotmr.reshape(9,100).mean(axis = 1) * 1e3, depths, 'b--', label = 'theorectical CH4')
plt.plot( [Cth*1e3]*9, depths, 'r-', label = 'threshold')
plt.legend(loc = 0)
plt.xlabel('CH4 concentration (mM)')
plt.ylabel('Depth (m)')

#%% Method 2: calculate the reaction rate of the CH4 removal reaction as the ebulliton flux, based on CH4 concentration

# Import the reaction rate parameters
Vmax = 5e-10      #maximum reaction rate, unit: mol L-1 s-1
Cth = 0.0014236   #threshold concentration, unit: mol L-1
f = 1e4/Cth       #scaling factor


I = np.arctan((ch4_tomorrow - Cth)*f) / 3.14159 + 0.5   #calculate the reaction rate inhibition term
R = Vmax * I         #the actual rate of ebullition, mol L-1 s-1
R_day = (R * 3600 * 24).reshape(900,1)   #convert to mol L-1 day-1

ch4_test = ch4_tomorrow - R_day


# now calculate the amount (mol) of CH4 removed by ebullition from the simulation domain
Volume = Thickness * Area * Porosity  #water volume of each of the 900 grid cells
Flux = np.sum((R_day * 1e3) * Volume)  #the amount of CH4 removed by ebullition from the simulation domain, unit: mol d-1
Flux = Flux / (0.1 * 0.1)   #divided by the surface area of the simulation domain to convert to mol m-2 d-1

# This method can overestimate the CH4 flux, because it assumes that the reaction proceeds for one day,
# but actually this reaction stops once the CH4 concentration falls below the threshold



#%% Method 3: OLD APPROACH, BASED ON A MASS BALANCE APPROACH

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

MetLoss = MetThry_sum - MetActl_sum   #tota
#methane loss, including diffusion via sed_air-interface and ebullition
MetLoss_rate = MetLoss / 0.01 *1e3        #methane loss rate, unit: mmmol/m2/day, 0.01 is the domain area (m2)

print(MetLoss_rate)

#%% the CH4 emission via diffusion
print(Mass[29,27]*100*1000)
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



#%% make the bar plot
import seaborn as sns
plt.rcParams.update({'font.size': 15})
sns.barplot(x = 'Group', y = 'Value', hue = 'Sorts', data = E_ch4)
plt.ylabel('Flux (mol m-2 day-1)')
plt.xlabel('')
plt.title('Ebullition flux')
plt.ylim(0,0.013)
plt.legend(loc = 0)


#%% only show the data with S cycling
plt.bar(1, 17.1517, 0.5, color = '#303030')
plt.bar(2, 15.1256, 0.5, color = '#24AEDB')
plt.bar(3, 14.9851, 0.5, color = '#D02F5E')
labels = ['No O2', 'Homogeneity', 'Heterogeneity']
plt.xticks(np.arange(1, 4, step = 1), labels)
plt.ylabel('CH4 Efflux (mmol m-2 d-1)')



#%% plot all three fluxes of three O2 injection modes together

subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
superscript = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

labels = ['No O2 release'.translate(subscript), 'Homogeneity', 'Heterogeneity']
NO = FluxCmpr[0,:]
Homo = FluxCmpr[1,:]
Het = FluxCmpr[2,:]
x = np.array([0, 2, 4, 6])
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(x - 0.3, NO, width, label = labels[0], color = '#303030')
rects2 = ax.bar(x, Homo, width, label = labels[1], color = '#24AEDB')
rects3 = ax.bar(x + 0.3, Het, width, label = labels[2], color = '#D02F5E')


xticklabs = []
plt.legend(loc = 0)
plt.ylabel('CH4 Flux'.translate(subscript) + ' (mmol m-2 d-1)'.translate(superscript))
plt.xticks(np.array([0, 2, 4, 6]), ['Total', 'Diffusion', 'Plant', 'Ebullition'])


#%% plot all three fluxes under different S levels
labels = ['no S', 'low S', 'medium S', 'high S']
x = np.array([0, 2, 4])
width = 0.2

noS = FluxCmpr[0,:]


fig, ax = plt.subplots()
rects1 = ax.bar(x - 0.4, noS, width, label = labels[0], color = '#303030')
rects2 = ax.bar(x - 0.2, np.array(MetFlux_lowS), width, label = labels[1], color = '#24AEDB')
rects3 = ax.bar(x + 0.2, np.array(MetFlux_medS), width, label = labels[2], color = 'y')
rects4 = ax.bar(x + 0.4, np.array(MetFlux_highS), width, label = labels[3], color = '#D02F5E')

xticklabs = []
plt.legend(loc = 0)
plt.ylabel('CH4 Flux (mmol m-2 d-1)')
plt.xticks(np.array([0, 2, 4]), ['Total Flux', 'Interface diffusion', 'Ebullition'])



#%% Convert data from Villa 2020 to parameterization of the Plant-mediated transport
conductance = 2.7e-3   #conductance of CH4, unit: m d-1
LAI = 2.78  #leaf area index, leaf area/ground area
porosity = 0.8
c_air = 0
c_soil = 1e-3   #CH4 concentration in soil porewater, unit: mol L-1
F_leafarea = conductance * (c_soil * 1e3 - c_air *1e3)   # the flux of CH4 dependent on the concentration gradient, unit: mol m-2(leaf area) d-1
F_groundarea = F_leafarea * LAI   #flux converted to per ground area, unit:mol m-2(ground) d-1

# convert the flux to concentration change in the soil cells
domain_area = 0.1 * 0.1  #surface area of the simulation domain
dz = 0.075   # thickness of the rooting zone
vol_rootzone = domain_area * dz * porosity   # water volume of the rooting zone
d_conc = F_groundarea * domain_area / vol_rootzone * 1e-3   # the change in CH4 concentration due to plant-mediated CH4 transport, unit: mol L-1 d-1
d_conc = d_conc / 24 / 3600   #change in CH4 concentration at [CH4]=1e-3 mol L-1, unit: mol L-1 s-1

v_max = d_conc * 10
HSC = c_soil * 10


# for heterogeneous setup, CH4 is removed only from the root cells
cell_area = 0.01 * 0.01
root_number = 20
vol_rootcells = cell_area * dz * root_number * porosity
d_conc_het = F_groundarea * domain_area / vol_rootcells * 1e-3
d_conc_het = d_conc_het / 24/ 3600

v_max_het = d_conc_het * 10
HSC_het = c_soil * 10