#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 06:59:54 2018

@author: huang
"""

import qutip as qt
import numpy as np

A = qt.Qobj(np.arange(4).reshape((2, 2)))
print(A)
print(qt.operator_to_vector(A))