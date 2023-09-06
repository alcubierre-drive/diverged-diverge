#!/usr/bin/env python3
#coding:utf8
import diverge.output as do
import matplotlib.pyplot as plt
import numpy as np

M = do.read('model.dvg')

fig = plt.figure(layout="constrained", figsize=(4,3))
xvals = np.concatenate( [[0], np.cumsum(np.sqrt((np.diff(M.bandstructure[:,1:],axis=0)**2).sum(axis=1)))] )
plt.plot( xvals, M.bandstructure[:,:-3], c='navy' )
plt.xticks( [xvals[0], xvals[300], xvals[600], xvals[900]], [r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$'] )
plt.xlim( xvals[0], xvals[-1] )
plt.ylabel( r'$\epsilon_b({\bf k})$' )
plt.show()
