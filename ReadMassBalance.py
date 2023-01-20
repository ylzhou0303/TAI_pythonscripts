# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:10:08 2022

@author: YZ60069
"""

#%% Opens the PFLOTRAN mass balance file
import pickle
import numpy as np
import pandas as pd

#%%
file_name = r"C:\Users\yz60069\TAI\TAI_fresh\TAI_wetlands-mas.dat"

#step 1: extract the headers
with open(file_name,'r') as inputFile:
    read_lines = inputFile.readlines()
   
variable_str = read_lines[0] #read in the 1nd line, getting variable names as strings
variable_list = variable_str.replace('"','').replace('\n','').split(',')
variable_list_cor = variable_list[0:7] + variable_list[8:]   #the 7th variable is always missing in the mass balance file


#step 2: extract the numbers
Mass = list()

for i in range(1, len(read_lines)):
    data_str = read_lines[i].replace(' -','  -').replace('    ','  ').split('  ')
    data_str = data_str[1:len(data_str)]
    Mass.append(np.array(data_str, dtype = np.float32))

Mass = np.array(Mass)

#create a data frame so it is easier to check
Mass_df = pd.DataFrame( data = Mass, columns = variable_list_cor)

#%%
filename = 'ExampleMassBalance.pickle'
with open(r'C:\MBL\Research\PFLOTRAN DATA\pflotran outputs\Example_MassBalanceFile' + filename, 'wb') as handle:
    pickle.dump([mass_bal, variable_list_cor], handle)

#%%
filename = 'ExampleMassBalance.pickle'
with open(r'C:\MBL\Research\PFLOTRAN DATA\pflotran outputs\Example_MassBalanceFile' + filename, 'rb') as handle:
    MassBalance = pickle.load(handle)