#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 21:07:40 2018

@author: huang
"""
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

def sigmoid(c):
    res = 1/(1 + np.exp(-c))
    if res > 0.5:
        return 1
    else:
        return -1

def perceptron():
    global df, cost,eta,x,w,lens,b,file_o
    count = 0
    for i in range(lens):
        x[0] = df['X1'][i]
        x[1] = df['X2'][i]
        y = df['y'][i]
        c = w*x + b
        #print(c)
        #For each instance the proceptron makes its predictions
        y1 = sigmoid(c)
        #print("predicted y1 and original y",y1,y)
        if y*y1<0:
            count +=1
            w = w + eta*x.T*y
            #print('w:',w)
            b = b + eta*y
            #print('b:',b)
            cost += -y*(w*x +b)
            #print('cost:',cost)
            #print(w[0,0],w[0,1],b)
    if count != 0:
        #Remember the comma
        file_o.write("{0:10.6f}, {1:10.6f}, {2:10.6f}\n".format(w[0,0],w[0,1],b))
        perceptron()
    else:
        file_o.write("{0:10.6f}, {1:10.6f}, {2:10.6f}\n".format(w[0,0],w[0,1],b))
        return

if __name__ == "__main__":
    input_name = sys.argv[1]  #To get the parameter 'input1.csv'
    output_name =sys.argv[2]
    df = pd.read_csv(input_name, names=['X1','X2','y'])
    #print(df)

    b = 0
    w = [1, 1]
    x = [0, 0]
    lens = df.iloc[:,0].size #Get the number of rows of DataFrame
    cost = 20

    w = np.matrix(w)
    x = np.matrix(x).T
    eta = 0.01

    #Then create a new file : output1.csv
    file_o = open(output_name, 'w') #the cube file as output
    perceptron()
    plt.figure()
    
    # To seperate the date with label = -1
    df1 = df[df['y']==-1]
    
    ax = df.plot.scatter(x='X1', y='X2', color='DarkBlue', label='1');
    df1.plot.scatter(x='X1', y='X2', color='DarkRed', label='-1', ax=ax);
    
    xx = np.linspace(0,17,100)
    yy = [-w[0,0]/w[0,1] * i - b/w[0,1] for i in xx]
    plt.plot(xx,yy)
    plt.show()