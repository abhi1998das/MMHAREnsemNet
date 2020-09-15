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



data_path = "/home/taichi10/Documents/college/ML/Skeleton"

#skel_paths = glob.glob(os.path.join(data_path, '*.npy'), recursive = True)

skel_paths = os.listdir(data_path)
   
print(skel_paths[0])


for path in tqdm.tqdm(skel_paths):
 
  print(path)
 
  filename = path.split('\\')[7]
  filenamesplit = filename.split('_')
  action = filenamesplit[2]
  subject = filenamesplit[1]
 
  #current = sio.loadmat(path)['d_skel']
  current = np.load(path)

  #current = np.transpose(current , (2,0,1))
  print(current.shape)
 
  numframes = current.shape[0]
  idxs = list(range(31))
  #print(idxs)
  idxs.remove(2)
  idxs.remove(4)
  idxs.remove(5)
  idxs.remove(12)
  idxs.remove(13)
  idxs.remove(19)
  idxs.remove(20)
  idxs.remove(24)
  idxs.remove(28)
  #print(idxs)
  current = current[:,idxs,:]
  print(current.shape)

 
 
  jadlist = []
 
  for a in range(numframes):
 
    xyz = current[a,:,:]
    jad = []
    cnt = 0
    for i in range(0,22):
      for j in range (0,i):
        for k in range(0,j):
         
          #print(i,j,k)
           
          pi = xyz[i,:]
          pj = xyz[j,:]
          pk = xyz[k,:]
         
          pij = pi - pj
          pjk = pj- pk
          pik = pi - pk
          pkj = pk - pj
          pji = pj- pi
         
          dotprodj = np.dot(pij,pjk)
          dotprodk = np.dot(pik,pkj)
          dotprodi = np.dot(pji,pik)
         
          normij = np.linalg.norm(pij)
          normjk = np.linalg.norm(pjk)
          normik = np.linalg.norm(pik)
         
          cosinei = np.cos( dotprodi / (normij * normik)  )
          cosinej = np.cos( dotprodj / (normij * normjk)  )
          cosinek = np.cos( dotprodj / (normik * normjk)  )
         
          ai = np.arccos(np.clip(cosinei, -1, 1))
          aj = np.arccos(cosinej)
          ak = np.arccos(cosinek)
         
          #print(ai,aj,ak)
         
          jad.append( ai )
          jad.append( aj )
          jad.append( ak )
         
                 
          #aijk = (np.linalg.norm(xyz[i,:]-xyz[j,:]))
          #jad.append( dij)
          #jadlist[a,cnt] = dij
          #cnt=cnt+1
 
 
    jadlist.append(jad)
   

 
  jadlist = np.array(jadlist)
  jadlist2=[]
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
