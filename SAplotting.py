# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:59:27 2023

@author: YZ60069
"""

#%% this file plot the results of sensitivity analysis
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots()

var_id = 5
var_str = Var_str[var_id]
t = 30

if var_id == 1:
    conv = 1/2.5e-4*100
else:
    conv = 1e6
# for i in range(0, ncols):
#     conc = Data_oxyhet[i:ngrids:ncols, t, var_id] * conv
#     plt.plot(conc, depths,color = 'skyblue', linestyle = '-')

# plt.plot(conc,depths, color = 'skyblue', linestyle = '-', label = 'O2 het, each column')

# calculate and plot mean profiles
MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_m[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs) * conv
plt.plot(MeanProfs[1:8,t], depths[1:8]*100, '-', color ='#303030', label = 'medium')  #convert depth to cm



MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_lo[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs) * conv
plt.plot(MeanProfs[1:8,t], depths[1:8]*100, '-', color = '#24AEDB', label = 'low')


MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_hi[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs) * conv
plt.plot(MeanProfs[1:8,t], depths[1:8]*100, '-', color = '#D02F5E', label = 'high')





subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
superscript = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

plt.legend(loc = 0)
#plt.xlabel(var_str[6:len(var_str)-4] + ' (μM)')

if var_id == 1:
    xlab = 'O2 saturation (%)'
elif var_id == 2:
    xlab = 'CH4(μM)'.translate(subscript)
elif var_id == 3:
    xlab = 'DOC (μM)'
elif var_id == 4:
    xlab = 'SO4'.translate(subscript) + '2-(μM)'.translate(superscript)
elif var_id == 5:
    xlab = 'H2S(aq) (μM)'.translate(subscript)

plt.xlabel(xlab)
plt.ylabel('Depth(cm)')





#%% plot results of inhibition term
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots()

var_id = 3
var_str = Var_str[var_id]
t = 30

if var_id == 1:
    conv = 1/2.5e-4*100
else:
    conv = 1e6


# calculate and plot mean profiles
MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_1[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs) * conv
plt.plot(MeanProfs[1:8,t], depths[1:8]*100, '-', color ='#303030', label = 'K_I = 0')  #convert depth to cm



MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_2[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs) * conv
plt.plot(MeanProfs[1:8,t], depths[1:8]*100, '-', color = '#24AEDB', label = 'K_I = 2.5e-4')


MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_3[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs) * conv
plt.plot(MeanProfs[1:8,t], depths[1:8]*100, '-', color = '#D02F5E', label = 'K_I = 1e-4')



MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_4[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs) * conv
plt.plot(MeanProfs[1:8,t], depths[1:8]*100, '-', color = 'g', label = 'K_I = 5e-5')

subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
superscript = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

plt.legend(loc = 0)
#plt.xlabel(var_str[6:len(var_str)-4] + ' (μM)')

if var_id == 1:
    xlab = 'O2 saturation (%)'
elif var_id == 2:
    xlab = 'CH4(μM)'.translate(subscript)
elif var_id == 3:
    xlab = 'DOC (μM)'
elif var_id == 4:
    xlab = 'SO4'.translate(subscript) + '2-(μM)'.translate(superscript)
elif var_id == 5:
    xlab = 'H2S(aq) (μM)'.translate(subscript)

plt.xlabel(xlab)
plt.ylabel('Depth(cm)')
