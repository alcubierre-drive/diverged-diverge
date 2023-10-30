#!/usr/bin/env python3
#coding:utf8
import diverge.output as do
import matplotlib.pyplot as plt
import numpy as np

# helper function to map a complex array to colors
def complex_cmap( ary ):
    A = ary.flatten()
    C = plt.cm.hsv( np.angle(A)/(2*np.pi)+0.5)
    C[:,3] *= np.abs(A) / np.abs(A).max()
    return C.reshape((*ary.shape, 4))

M = do.read('honeycomb.cpp.mod.dvg')
O = do.read('honeycomb.cpp.out.dvg')

K = O.P_qV[0][1]
vector_colors = complex_cmap( O.P_qV[0][3][1:3] )
# select eigenvectors [1:3] and map to colors

fig, axs = plt.subplots( 2, 4, sharex=True, sharey=True,
                         layout="constrained", figsize=(3,3) )
for idx in range(2):
 for o1 in range(2):
  for o2 in range(2):
    ax = axs[o1,idx*2+o2]
    ax.scatter( *M.kmesh[K,:2].T, c=vector_colors[idx,:,o1,o2], s=70, lw=0 )
    ax.set_aspect(1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title( r'$\Delta^{%i}_{%i%i}({\bf k})$' % (idx+1,o1+1,o2+1),
                  fontsize=9 )
fig.savefig( 'patch.pdf' )
