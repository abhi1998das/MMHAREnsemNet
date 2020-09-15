#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 04:07:23 2020

@author: taichi10
"""


import numpy as np
import os

from PIL import Image
import glob
import tqdm
import math
import time
from pathlib import Path
from sys import exit



data_path = "/home/taichi10/Documents/college/ML/Signal"


  for i in range (jadlist-1):
      jadlist2.append(jadlist[i+1]-jadlist[i])


  #maximum = np.max(jadlist)
  #minimum = np.min(jadlist)  
 
 
  for i in range(jadlist2.shape[0]):
    maximum = np.max(jadlist2[i])
    minimum = np.min(jadlist2[i])  
    #print(maximum,minimum)
   
    jadlist2[i,:] = np.floor( (jadlist2[i,:] - minimum) / (maximum -minimum)  * (255.0-0) )
 
  print(jadlist.shape)
  #print(jadlist)
  jadlist2 = imresize(jadlist2,(265,4620),interp='bicubic')
  im = Image.fromarray(jadlist2)
 
  #filepath = path
  trainnames = ['bd' , 'mm' ]
 
  if subject in trainnames:
    filepath = r'C:/Users/AVI/Desktop/HDM05/trainang/'
  else:
    filepath = r'C:/Users/AVI/Desktop/HDM05/valang/'
       
  filedir = filepath + action
  if not os.path.exists(filedir):
    os.mkdir(filedir)
 
  filepath = filepath + action + r'/' + filename.replace('.npy','.jpg')
 
 
 
  im.save(filepath)