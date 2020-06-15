#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 07:15:05 2018

@author: huang
"""
import qutip as qt
import numpy as np

H = 10 * qt.sigmaz()

c1 = qt.destroy(2)
L = qt.liouvillian(H, [c1])

print(L)

S = (12 * L).expm()
print("S: ",S)