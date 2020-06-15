#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 07:06:18 2018

@author: huang
"""
import qutip as qt
import numpy as np

X = qt.sigmax()
S = qt.spre(X) * qt.spost(X.dag()) # Represents conjugation by X.
print(X)
print(S)
S2 = qt.to_super(X)
print(S2) # type of S is super

# Check if S is completely positive
print(S.iscp)