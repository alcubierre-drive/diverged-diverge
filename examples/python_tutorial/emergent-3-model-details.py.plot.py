#!/usr/bin/env python3
#coding:utf8
import diverge.output as do
import matplotlib.pyplot as plt

M = do.read('model.dvg')

fig = plt.figure(layout="constrained", figsize=(3,2))
xvals = do.bandstructure_xvals(M)
plt.plot( xvals, do.bandstructure_bands(M), c='navy' )
plt.xticks(do.bandstructure_ticks(M), [r'$\Gamma$',r'$M$',r'$X$',r'$\Gamma$'])
plt.xlim( xvals[0], xvals[-1] )
plt.ylabel( r'$\epsilon_b({\bf k})$' )
fig.savefig('bands.pdf')
