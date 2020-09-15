#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 23:06:44 2020

@author: taichi10
"""


import os
import scipy.io
import numpy as np
import math

import numpy

def smooth(x,window_len=11,window='flat'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """ 
     
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")
        

    if window_len<3:
        return x
    
    
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    

    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')
    
    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y    

    
    
    
from numpy import *
from pylab import *
    
def smooth_demo():    
    
    t=linspace(-4,4,100)
    x=sin(t)
    xn=x+randn(len(t))*0.1
    y=smooth(x)
    
    ws=31
    
    subplot(211)
    plot(ones(ws))
    
    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    
    hold(True)
    for w in windows[1:]:
        eval('plot('+w+'(ws) )')
    
    axis([0,30,0,1.1])
    
    legend(windows)
    title("The smoothing windows")
    subplot(212)
    plot(x)
    plot(xn)
    for w in windows:
        plot(smooth(xn,10,w))
    l=['original signal', 'signal with noise']
    l.extend(windows)
    
    legend(l)
    title("Smoothing a noisy signal")
    show()

    

files=os.listdir("/home/taichi10/Documents/college/ML/Inertial")

ma=-1
mi=10000

train=[]
label=[]




for file in files:
    pat="/home/taichi10/Documents/college/ML/Inertial/"+file
    t_str=pat.split("/")[7].split("_")[0]
    ty=""
    for i in range(1,len(t_str)):
        ty+=t_str[i]
    label.append(int(ty))
    
    mat = scipy.io.loadmat(pat)
    data=mat['d_iner']
    data=np.array(data)
    le=data.shape[0]
    tex=[]
    tey=[]
    tez=[]
    tem=[]
    train_fet=[]
    for l in range(le):
        tex.append(data[l][3])
        tey.append(data[l][4])
        tez.append(data[l][5])
        tem.append(math.sqrt(data[l][0]*data[l][0]+data[l][1]*data[l][1]+data[l][2]*data[l][2]))
    tex=np.array(tex)
    texs=smooth(tex,3)
    tey=np.array(tey)
    teys=smooth(tey,3)
    tez=np.array(tez)
    tezs=smooth(tez,3)
    tem=np.array(tem)
    tems=smooth(tem,3)
    
#    --------------x----------------------------------------
    no=texs.shape[0]
    tmu=0
    for i in range(no):
        tmu+=texs[i]
    mux=tmu/no
    tmu=0
    for i in range(no-1):
        tmu+=abs(texs[i+1]-texs[i])
    mux1=tmu/no
    tmu=0
    for i in range(no-2):
        tmu+=abs(texs[i+2]-2*texs[i+1]+texs[i])
    mux2=tmu/no
    train_fet.append(mux)
    train_fet.append(mux1)
    train_fet.append(mux2)
#    --------------y----------------------------------------
    no=teys.shape[0]
    tmu=0
    for i in range(no):
        tmu+=teys[i]
    muy=tmu/no
    tmu=0
    for i in range(no-1):
        tmu+=abs(teys[i+1]-teys[i])
    muy1=tmu/no
    tmu=0
    for i in range(no-2):
        tmu+=abs(teys[i+2]-2*teys[i+1]+teys[i])
    muy2=tmu/no
    train_fet.append(muy)
    train_fet.append(muy1)
    train_fet.append(muy2)
#    --------------z----------------------------------------
    no=tezs.shape[0]
    tmu=0
    for i in range(no):
        tmu+=tezs[i]
    muz=tmu/no
    tmu=0
    for i in range(no-1):
        tmu+=abs(tezs[i+1]-tezs[i])
    muz1=tmu/no
    tmu=0
    for i in range(no-2):
        tmu+=abs(tezs[i+2]-2*tezs[i+1]+tezs[i])
    muz2=tmu/no
    train_fet.append(muz)
    train_fet.append(muz1)
    train_fet.append(muz2)

#    --------------m----------------------------------------
    no=texs.shape[0]
    tmu=0
    for i in range(no):
        tmu+=tems[i]
    mum=tmu/no
    tmu=0
    for i in range(no-1):
        tmu+=abs(tems[i+1]-tems[i])
    mum1=tmu/no
    tmu=0
    for i in range(no-2):
        tmu+=abs(tems[i+2]-2*tems[i+1]+tems[i])
    mum2=tmu/no
    train_fet.append(mum)
    train_fet.append(mum1)
    train_fet.append(mum2)
    
    train_fet=np.array(train_fet)
    train.append(train_fet)
    
 

    
train=np.array(train)
label=np.array(label)

print(train.shape)
print(label.shape)

np.save('train_gyro.npy',train)
np.save('label_gyro.npy',label)



    
