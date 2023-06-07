# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:20:36 2023

@author: YZ60069
"""


from contextlib import AsyncExitStack
import pandas as pd
#from socket import AF_X25
import numpy as np
import pickle
import statistics as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import os
os.chdir('C:/Users/yz60069/TAI/TAI_fresh')
#os.chdir('C:/MBL/Research/PFLOTRAN DATA/pflotran outputs/OxyHet/Marsh Interior/All inundated/ root')
import scipy.io
from scipy.stats import pearsonr