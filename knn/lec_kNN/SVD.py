#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 10:14:43 2018

@author: huang
"""

from numpy import *  
   
def loadData():  
    return [[1,2,3],  
             [4,5,6],  
             [7,8,9],  
             [10,11,12]] 
  
data=loadData()  
  
u,sigma,vt=linalg.svd(data)  
  
print("sigma=",sigma)  
  
sig3=mat([[sigma[0],0,0],  
      [0,sigma[1],0],  
      [0,0,sigma[2]]])  
 
print(u[:,:3]*sig3*vt[:3,:])