# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:18:04 2023

@author: YZ60069
"""

# This file calculates the mass balance of O2
# to investigate the balance of O2 supply and consumption

fluid_rate = 0.05e-8           # flow rate of O2 injection, m3/hr
fluid_conc = 0.7               # concentration of O2, mol/L

Oxy_in = fluid_rate / 3600 * (fluid_conc * 1e3)  #oxygen input rate to the root cell, mol s-1

cell_id = 326
t = 30
Oxy_consump = Rates[cell_id,t,0] + 2 * Rates[cell_id,t,2] + 2 * Rates[cell_id,t,4]
#oxygen consumption rate of DOM aerobic respiration, CH4 oxidation and H2S oxidaiton
# 1 mol of CH4 oxdiation consumes 2 mols of O2; 1 mol H2S oxidation consumes 2 mols of O2
# unit: mol L-1 s-1

cell_vol = 0.01 * 0.01 * 0.075 * Porosity #volume of the water in the root cell, unit: m3, DON'T FORGET THE FUCKING POROSITY!!!!
Oxy_consump = Oxy_consump * 1e3 * cell_vol    #unit:mol s-1


# the actual change in oxygen amount in that cell
Oxy_inc = (Full_Data[cell_id,t,1] - Full_Data[cell_id,t-1,1]) * 1e3 * cell_vol /(3600*24)   #unit:mol s-1


#%% calculate the diffusion, J = -D*(dc/dx)
dc = (Full_Data[327,t,1] - Full_Data[326,t,1]) * 1e3 #difference in O2 concentration between the neighboring cells, unit:mol m-3
dx = 0.01           #distance between the center point of the two cells, which is 1cm, unit:m
D = 1.3e-9          #diffusion coefficient, unit: m2 s-1
time = 3600 * 24       #the time interval between the two measurement time points, unit: s

J = D*(dc/dx) / time     #flux through a unit surface per unit of time, unit:mol m-2 s-1
