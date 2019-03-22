#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:24:29 2019

@author: buchholz
"""

import pandas as pd
import re
from scipy.io import loadmat
import numpy as np

all_files = loadmat('file_list.mat')
test = loadmat('test_list.mat')
train = loadmat('train_list.mat')

all_files = pd.DataFrame(np.hstack((all_files['file_list'])),columns=['file'])
test= pd.DataFrame(np.hstack((test['file_list'])),columns=['file'])
train= pd.DataFrame(np.hstack((train['file_list'])),columns=['file'])

from shutil import copyfile
import os
for file in test['file']:
  
  filename = str('test/'+file[0])
  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))
  copyfile(str('Images/'+file[0]),str('test/'+file[0]))

from shutil import copyfile
import os
for file in train['file']:
  filename = str('train/'+file[0])
  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))
  copyfile(str('Images/'+file[0]),str('train/'+file[0]))
