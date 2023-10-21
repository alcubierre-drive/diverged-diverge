#!/usr/bin/env python3
#coding:utf8
import diverge.output as do
import matplotlib.pyplot as plt
import numpy as np

M = do.read('model.dvg')

fig = plt.figure(layout="constrained", figsize=(4,3))
x = do.bandstructure_xvals(M)
plt.plot( x, do.bandstructure_bands(M), color='navy' )
plt.xticks( do.bandstructure_ticks(M), [r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$'] )
plt.xlim( x[0], x[-1] )
plt.ylabel( r'$\epsilon_b({\bf k})$' )
plt.show()
