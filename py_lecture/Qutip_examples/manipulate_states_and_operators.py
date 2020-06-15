#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 01:14:42 2018

@author: huang
"""
import qutip as qt
import numpy as np

vac = qt.states.basis(5, 0)
print(vac)

a = qt.operators.destroy(5)
print(a)
print(a*vac)

a1 = a.dag() * vac
print(a1)
a2 = a.dag() * a.dag() * a.dag() * a.dag() * a.dag() * vac
a3 = a.dag() * a.dag() * a.dag() * (a.dag() * vac).unit()
print("a2: ",a2)
print("a3: ",a3)

# another methond instead of using c * a * (c ** 2 * vac).unit()
n = qt.operators.num(5)
print(n)
ket = qt.basis(5, 2)
print(n * ket)
