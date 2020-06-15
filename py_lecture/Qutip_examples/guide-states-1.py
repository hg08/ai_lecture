#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 07:15:05 2018

@author: huang

For qubits, a particularly useful way to visualize superoperators is 
to plot them in the Pauli basis, such that Sμ,ν=⟨⟨σμ|S[σν]⟩⟩. Because 
the Pauli basis is Hermitian, Sμ,ν is a real number for all 
Hermitian-preserving superoperators S, allowing us to plot the elements 
of S as a Hinton diagram. In such diagrams, positive elements are 
indicated by white squares, and negative elements by black squares. 
The size of each element is indicated by the size of the corresponding 
square. For instance, let S[ρ]=σxρσ†xS[ρ]=σxρσx†.  
We can quickly see this by noting that the Y and Z elements of the 
Hinton diagram for S are negative:

"""



import qutip as qt

qt.settings.colorblind_safe = True

import matplotlib.pyplot as plt
plt.rcParams['savefig.transparent'] = True

X = qt.sigmax()
S = qt.spre(X) * qt.spost(X.dag())

print(S)
qt.hinton(S)