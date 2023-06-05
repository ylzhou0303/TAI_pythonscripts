# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:08:22 2023

@author: YZ60069
"""

# This script calculates the mass balance for the flow condition in PFLOTRAN model
# Tracer2 mass balance, Tracer2 is used as an inhibition species in the reaction that simulates plant-mediated CH4 transport
cell_vol = 0.01*0.01*0.075*0.8
flow_rate = 1e-8 / 100  #the liquid rate into each cell, unit: m3/hr
# how much of liquid is injected to each cell after 30 days?
flow_amount = flow_rate * 24 * 30
# it is equivalent to how much of the water volume in each cell?
perc = flow_amount/cell_vol    #0.8 is the porosity




#%% injection of Tracer2

Tracer2 = 1   #concentration of tracer2 in the injected liquid, 1 mol/L
#tracer2 injection rate, unit: mol L-1 s-1
Tracer2_rate = Tracer2 * 1e3 * flow_rate / 3600 / (cell_vol*1e3)  #unit: mol L-1 s-1