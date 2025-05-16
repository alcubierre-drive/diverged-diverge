#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import diverge.output as do
import fancymaps as fm
import matplotlib.pyplot as plt
import os
os.system( 'make; ./honeycomb.x' )

ccmap = fm.complex_cmap_factory( fm.Cycl_wdiv4 )
# ccmap = fm.complex_cmap_factory( plt.cm.hsv )

M = do.read( 'el_int.dvg' )
P = do.read( 'elph_post.dvg' )
pp = do.read( 'ph.dvg' )
plt.figure()
plt.plot( do.bandstructure_xvals(pp), do.bandstructure_bands(pp) )
plt.xticks( do.bandstructure_ticks(pp), 'GMKG' )

g = np.fromfile('g.bin', dtype=complex).reshape((np.prod(M.nk), 20, -1))

tumap = P.mi_to_tuffidx
formfactors = M.tu_ff[tumap]

fsel = formfactors['R'] > M.nk[0]//2
formfactors['R'][fsel] -= M.nk[0]

cg = ccmap(g)[0]
print( np.abs(g[0]).sum(axis=-2) )

fig, axs = plt.subplots( M.n_orb, M.n_orb*2, layout="constrained", sharex=True, sharey=True )
for o in range(M.n_orb):
 for p in range(M.n_orb):
  for xyz in range(2):
   ax = axs[o,p*2+xyz]
   ax.set_aspect(1)
   for iff, ff in enumerate(formfactors):
    of = ff['oto']
    if of == o:
        bond = ff['R'] @ M.lattice + M.positions[ff['ofrom']] - M.positions[ff['oto']]
        vecs = np.array([np.zeros_like(bond), bond])
        ax.plot( vecs[:,0], vecs[:,1], c=cg[iff,p*3+xyz], lw=5 )
        ax.scatter( *bond[:2], c='r', s=8, zorder=4 )
        ax.scatter( *(M.positions[p]-M.positions[o])[:2], c='k', s=40, zorder=3 )

# fig, axs = plt.subplots( M.n_orb, 1, layout="constrained", sharex=True, sharey=True )
# cg = ccmap( P.Dvec[0,0,0] )
# for o in range(M.n_orb):
#     ax = axs[o]
#     ax.set_aspect(1)
#     for iff, ff in enumerate(formfactors):
#         of = ff['oto']
#         if of == o:
#             bond = ff['R'] @ M.lattice + M.positions[ff['ofrom']] - M.positions[ff['oto']]
#             vecs = np.array([np.zeros_like(bond), bond])
#             ax.plot( vecs[:,0], vecs[:,1], c=cg[iff], lw=5 )
#             if np.isclose(np.linalg.norm(bond),0):
#                 ax.scatter( *bond[:2], color=cg[iff], s=80, zorder=4 )
#             else:
#                 ax.scatter( *bond[:2], color='gray', s=8, zorder=4 )

fig = plt.figure( layout="constrained" )
ax = fig.add_subplot( 111 )
for i in range(2):
 for j in range(2):
  ax.scatter(*(M.rlattice[0,:2]*i + M.rlattice[1,:2]*j + M.kmesh[:,:2]).T,
             c=np.log10(P.Dval[P.Doff]), cmap='nipy_spectral_r' )
ax.set_aspect(1)
plt.show()

# vals = g[0,0]
# # plt.scatter( 
# print( formfactors )

# plt.imshow( g[:,:,0,0].real )
# plt.imshow( np.abs( g.transpose((2,0,3,1)).reshape((24*20,-1)) ) )
# plt.colorbar()
# plt.imshow( np.real( g.transpose((2,0,3,1)).reshape((24*20,-1)) ) )
# plt.show()
