# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 09:03:42 2023

@author: YZ60069
"""

# This file calculates the respective contribution of different pathways of O2 impacts on CH4
# root O2 release impacts CH4 cycling via three pathways:
# 1) enhance CH4 oxidation
# 2) inhibit methanogenesis
# 3) enhance H2S oxidation to produce more SO42-, so enhance sulfate reduction and reduce the OM substrate, so reduce methanogenesis

import numpy as np
#%% Step1: calculate the net effect of root o2 release
# calculate the rate of net CH4 production for the NO mode (no oxygen)

#%% import results of the NO mode
# calculate the net production rate of CH4, subtract CH4 oxidation rate from methanogenesis rate
Rates_NO = Rates
Full_Data_NO = Full_Data
Met_NO = Rates_NO[300:400,30,1]  #extract the methanogenesis rate of all cells at the root depth on day30
MetOxi_NO = Rates_NO[300:400,30,2] #extract the methane oxidaiton rate of all cells at the root depth on day30

MetProd_NO = np.mean(Met_NO - MetOxi_NO)


#%% import results of the HetO mode
Rates_HetO = Rates
Full_Data_HetO = Full_Data
Met_HetO = Rates_HetO[300:400,30,1]  #extract the methanogenesis rate of all cells at the root depth on day30
MetOxi_HetO = Rates_HetO[300:400,30,2] #extract the methane oxidaiton rate of all cells at the root depth on day30

MetProd_HetO = np.mean(Met_HetO - MetOxi_HetO)

#%% The total effect of O2 release on net CH4 production rate
Total = MetProd_NO - MetProd_HetO   #the reduction of net CH4 production rate caused by root O2 release


#%% Step2： calculate the effect of O2 via CH4 oxidation
Effect_MetOxi = np.mean(MetOxi_HetO - MetOxi_NO)


#%% Step3:calculate the effect of O2 via inhibition of methanogenesis
o2_HetO = Full_Data_HetO[300:400,30,1]   #O2 concentration of all cells at the root depth on day30
Inhb_HetO = 2.5e-4 / (2.5e-4 + o2_HetO)  #Monod inhibition term of O2 for methanogenesis

o2_NO = Full_Data_NO[300:400,30,1]
Inhb_NO = 2.5e-4 / (2.5e-4 + o2_NO)

Effect_Inhb = np.mean((Met_HetO * (1 / Inhb_HetO - 1)) - (Met_NO * (1 / Inhb_NO - 1))) #the effect caused by enhancement of O2 inhibition of methanogenesis


#%% Step4: the effect of O2 via H2S oxidation
Effect_SulOxi = np.mean(Total - Effect_MetOxi - Effect_Inhb)














# Vmax_MetProd = 5e-10   #methane production maximum rate
# HSC_MetProd_dom = 2e-3  #half saturation concentration of DOM for methanogenesis reaction
# K_I_o2 = 2.5e-4   #half saturation concentration of O2 inhibition for methanogenesis

# Vmax_MetOxi = 4e-9  #maximum reaction rate of methane oxidation
# HSC_MetOxi_o2 = 1e-4  #half saturation concentration of O2 for methane oxidation
# HSC_MetOxi_ch4 = 3e-4 #half saturation concentration of CH4 for methane oxidation

# dom = Full_Data[300:399,30,3] #DOM concentration of all cells at the root layer depth at day 30
# o2 = Full_Data[300:399,]
# Monod = HSC_MetProd_dom / 

# R_MetProd = 

