#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import diverge.output as do
import fancymaps as fm
import matplotlib.pyplot as plt
import os

ccmap = fm.complex_cmap_factory( fm.Cycl_wdiv4 )
# ccmap = fm.complex_cmap_factory( plt.cm.hsv )

os.system( 'cd ..; make; cd phonon_debugging; PHONON_CFG=4 ./honeycomb.x' )

M = do.read( 'el_int.dvg' )
P = do.read( 'elph_post.dvg' )
pp = do.read( 'ph.dvg' )

PLOT_BANDS = False
PLOT_VERTEX = True

if PLOT_BANDS:
    plt.figure()
    plt.plot( do.bandstructure_xvals(pp), do.bandstructure_bands(pp) )
    plt.xticks( do.bandstructure_ticks(pp), 'GMKG' )

n_O = pp.n_orb
n_orbff = P.n_orbff
n_orb = M.n_orb

g = np.fromfile('g.bin', dtype=complex).reshape((np.prod(M.nk), n_orbff, n_O)) # (q,off,mode)
U = pp.U.reshape( (np.prod(pp.nk), np.prod(pp.nkf), n_O, n_O) )

tumap = P.mi_to_tuffidx
formfactors = M.tu_ff[tumap]
fsel = formfactors['R'] > M.nk[0]//2
formfactors['R'][fsel] -= M.nk[0]

qpt = M.nk[0]//2
modes = U[qpt,0].reshape((n_O, n_orb, 3)) # b,o,x
# print( modes )
cg = ccmap(g)[qpt]
# print( np.abs(g[qpt]).sum(axis=-2) )

fig, axs = plt.subplots( n_orb, n_O, layout="constrained", sharex=True, sharey=True )
for ofrom in range(n_orb):
    for mode in range(n_O):
        ax = axs[ofrom,mode]
        ax.set_aspect(1)
        for iff, ff in enumerate(formfactors):
            oto = ff['oto']
            ofrom_ = ff['ofrom']
            if ofrom_ != ofrom: continue
            v_to = -ff['R'] @ M.lattice + M.positions[oto]
            v_from = M.positions[ofrom]
            v_ = np.array( [v_from, v_to] )
            ax.plot( v_[:,0], v_[:,1], c=cg[iff,mode], lw=5 )
            ax.scatter( *v_from[:2], s=8, zorder=2, c='k' )
        if np.linalg.norm(modes[mode,:,:2].real) > 0.5:
            ax.quiver( *M.positions[:,:2].T, modes[mode,:,0].real, modes[mode,:,1].real, zorder=3, scale=5 )
        # ax.scatter( *bond[:2], c='r', s=8, zorder=4 )
        # ax.scatter( *(M.positions[p]-M.positions[o])[:2], c='k', s=40, zorder=3 )

if PLOT_VERTEX:
    Chans = [P.Pval[P.Poff], P.Cval[P.Coff], P.Dval[P.Doff]]
    for ichan, chan in enumerate(Chans):
        fig = plt.figure( layout="constrained" )
        ax = fig.add_subplot( 111 )
        for i in range(2):
         for j in range(2):
          im = ax.scatter(*(M.rlattice[0,:2]*i + M.rlattice[1,:2]*j + M.kmesh[:,:2]).T,
                     c=np.log10(chan), cmap='nipy_spectral_r' )
        plt.colorbar(im)
        fig.suptitle(str(ichan))
        ax.set_aspect(1)


plt.show()

