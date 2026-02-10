#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import fancymaps as fm
import diverge.output as do

ccmap = fm.complex_cmap_factory( fm.Cycl_wdiv4 )
P = do.read('tu_rpa_honey.c.out.dvg')
M = do.read('tu_rpa_honey.c.mod.dvg')

V = P.rpa_vecs[:,:,0,:,0]
E = P.rpa_vals
kidx = P.rpa_kidx
nval = E.size

for i in range(2):
    fig, axs = plt.subplots( 2, 2, sharex=True, sharey=True )
    cV = ccmap(V[i])
    k = []
    cVV = []
    for ii in range(-1,1):
        for jj in range(-1,1):
            k.append( M.rlattice[0:1]*ii + M.rlattice[1:2]*jj + M.kfmesh[kidx] )
            cVV.append( cV )
    rn = np.linalg.norm(M.rlattice[0])
    k = np.concatenate(k)
    cVV = np.concatenate(cVV)
    for o in range(2):
        for p in range(2):
            axs[o,p].scatter( *k[:,:2].T, c=cVV[:,o,p] )
            axs[o,p].set_aspect(1)
            axs[o,p].set_xlim(-0.61*rn, 0.61*rn)
            axs[o,p].set_ylim(-0.61*rn, 0.61*rn)
    fig.suptitle(f'{E[i]:.2f}')
plt.show()

