# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:08:31 2023

@author: YZ60069
"""

#%% This file calculates the flux of CH4 
# I set up artificial reactions in PFLOTRAN to simulate the ebullition process and the plant-mediated transport of CH4
# In those two artificial reactions, CH4 is transformed to tracers, so this file calcualtes the daily increase in the tracers as the flux

#%% 1. Plant-mediated CH4 flux
# I used Tracer1 in PFLOTRAN to keep track of the plant-mediated flux
# So the amount change in Tracer1 from day9 to day10 is the plant-mediated flux

# Specify the column number of Tracer1 in the mass balance file
Tracer1_col = 9       # the column number of the total amount of Tracer1 within the entire domain (mol) in the mass balance file
bc_col = 34          # the column number of the exchange of Tracer1 across the boundary (mol/d) in the mass balance file
day10_row = 9       # the row number of data of day9
day9_row = 8        # row number of data of day10

#increase in Tracer1 in the entire modeled soil column from day9 to day10, the amount of Tracer1 left via the boundary needs to be added back in
PlantF_domain = Mass[day10_row,Tracer1_col] - Mass[day9_row,Tracer1_col] - Mass[day9_row,bc_col]

# Convert to an areal flux
PlantF_areal = PlantF_domain * 1000 / 0.01   #mmol/m2/day



#%% 2. Calculate the ebullition flux
# I used Tracer6 to keep track of the ebullition
Tracer6_col = 13    # Global tracer6 (mol) amount is on the 13th column of the mass balance file
bc_col = 38        # daily exchange rate of tracer6 across the boundary is on the 38th column of the mass balance file

# The increase in Tracer6 amount in the entire simulation domain from day9 to day10 would be the daily ebullition flux
Ebl_domain = Mass[day10_row, Tracer6_col] - Mass[day9_row, Tracer6_col] - Mass[day9_row, bc_col]

# convert to an areal flux
Ebl_areal = Ebl_domain * 1000 / 0.01      #mmol m-2 d-1


#%% Compile fluxes to calculate the total flux
MetF_Diffusion = - Mass[8,39]*1e3/0.01   # the diffusion flux via sediment-air interface, unit: mmol m-2 d-1

MetF_total = MetF_Diffusion + Ebl_areal + PlantF_areal   #total flux

MetF = {'Total Flux': MetF_total, 'Diffusion': MetF_Diffusion, 'Plant-mediated': PlantF_areal, 'Ebullition': Ebl_areal}
MetF = pd.DataFrame(MetF, index = ['Flux'])
MetF_np = np.array(MetF)
print(MetF)



