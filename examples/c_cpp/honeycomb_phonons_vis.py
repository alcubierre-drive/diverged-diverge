#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import diverge.output as do
import fancymaps as fm
import matplotlib.pyplot as plt
import os
os.system( 'make; ./honeycomb_phonons.x' )

ccmap = fm.complex_cmap_factory( fm.Cycl_wdiv4 )
# ccmap = fm.complex_cmap_factory( plt.cm.hsv )

M = do.read( 'el_int.dvg' )
P = do.read( 'elph_post.dvg' )
pp = do.read( 'ph.dvg' )
plt.figure()
plt.plot( do.bandstructure_xvals(pp), do.bandstructure_bands(pp) )
plt.xticks( do.bandstructure_ticks(pp), 'GMKG' )

g = np.fromfile('g.bin', dtype=complex).reshape((24*24, 20, -1))

tumap = P.mi_to_tuffidx
formfactors = M.tu_ff[tumap]

fsel = formfactors['R'] > M.nk[0]//2
formfactors['R'][fsel] -= M.nk[0]

cg = ccmap(g)[0]
print( np.abs(g).sum() )

modes = pp.U[0]
xg = np.einsum( 'qfO, bO -> qbf', g, modes )
print( np.abs(xg[0]).sum(axis=-1) )

fig, axs = plt.subplots( M.n_orb, 2, layout="constrained", sharex=True, sharey=True )
for p in range(M.n_orb):
    for xyz in range(2):
        for iff, ff in enumerate(formfactors):
            bondvec = ff['R'] @ M.lattice + M.positions[ff['ofrom']]
            start = M.positions[ff['oto']]
            vecs = np.array([start, bondvec])
            axs[p,xyz].plot( vecs[:,0], vecs[:,1], c=cg[iff,p*3+xyz], lw=5 )
            axs[p,xyz].scatter( *vecs[:,:2].T, c='r', s=8, zorder=4 )
        axs[p,xyz].scatter( *M.positions[p,:2], c='k', s=40, zorder=3 )
        axs[p,xyz].set_aspect(1)
plt.show()

# vals = g[0,0]
# # plt.scatter( 
# print( formfactors )

# plt.imshow( g[:,:,0,0].real )
# plt.imshow( np.abs( g.transpose((2,0,3,1)).reshape((24*20,-1)) ) )
# plt.colorbar()
# plt.imshow( np.real( g.transpose((2,0,3,1)).reshape((24*20,-1)) ) )
# plt.show()
