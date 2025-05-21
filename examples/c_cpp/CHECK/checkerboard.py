#!/usr/bin/env python3
# coding: utf-8

import diverge.output as do
import matplotlib.pyplot as plt
import numpy as np
import fancymaps as fm

import os
os.system( 'make' )
os.system( './checkerboard.x' )

EF = 1.0
M = do.read('checkerboard.c_mod.dvg')

Berry = M.data.view( dtype=np.float64 ).reshape( (*(M.nk*M.nkf), M.n_spin*M.n_orb, 3) )[:,:,0,:,-1]
plt.imshow( Berry[:,:,0], cmap=fm.bdiv2 )
b0 = Berry[...,0]
plt.title( b0[b0 > 0].sum() )
plt.show()

kidx = []
spin = []
pauli_z = np.kron( np.array([[1,0],[0,-1]]), np.eye(2) )
for k in range(len(M.kfmesh)):
    ww = np.where( np.abs(M.E[k] - EF) < 4e-2 )
    if len(ww[0]) > 0:
        kidx.append( k )
        if M.n_spin > 1:
            bloch = M.U[k,ww[0]]
            bloch_mat = bloch @ pauli_z @ bloch.T.conj()
            s = 0.0
            for b in range(len(bloch)):
                s += bloch_mat[b,b]
            spin.append( s.real )
        else:
            spin.append( 0.0 )

plt.imshow( M.E[:,0].reshape((40*15,-1)), cmap='nipy_spectral_r' )
plt.figure()
# plt.scatter( *FS_k[:,:2].T, c='k', s=1, lw=0 )
plt.scatter( *M.kfmesh[kidx,:2].T, c=spin, cmap=fm.bdiv2, vmin=-1, vmax=1, s=1, lw=0 )
# plt.scatter( *(FS_k + 0.5*M.rlattice[0] + 0.5*M.rlattice[1])[:,:2].T, c='r', s=1, lw=0, alpha=0.4 )
plt.gca().set_aspect(1)
plt.figure()
plt.plot( do.bandstructure_xvals(M), do.bandstructure_bands(M), c='r' )
plt.xticks( do.bandstructure_ticks(M), 'GXMG' )
plt.show()
