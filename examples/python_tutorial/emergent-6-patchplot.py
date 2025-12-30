#!/usr/bin/env python3
#coding:utf8
import diverge.output as do
import matplotlib.pyplot as plt
import numpy as np

from cycler import cycler
default_cycle = cycler(color=plt.cm.tab20( np.linspace(0,1,20) ))
plt.rc( "axes", prop_cycle=default_cycle )

M = do.read('model_npatch.dvg')

fig = plt.figure(layout="constrained", figsize=(4,4))

for i in range(len(M.patches)):
    refined = M.p_map[M.p_displ[i]:M.p_displ[i]+M.p_count[i]]
    patch = M.patches[i]

    actual_idx_y = (patch%M.nk[1] + refined%M.nk[1])%M.nk[1]
    actual_idx_x = (patch//M.nk[1] + refined//M.nk[1])%M.nk[1]

    actual_idx = actual_idx_y + actual_idx_x*M.nk[1]
    plt.scatter( *M.kmesh[actual_idx,:2].T, s=1 )
    plt.scatter( *M.kmesh[patch,:2], s=10, c='k' )
plt.xticks( [0,np.pi,2*np.pi], [r'$0$', r'$\pi$', r'$2\pi$'] )
plt.yticks( [0,np.pi,2*np.pi], [r'$0$', r'$\pi$', r'$2\pi$'] )
plt.xlim( 0,2*np.pi )
plt.ylim( 0,2*np.pi )
plt.gca().set_aspect(1)
plt.title( 'patching & refinement' )
plt.show()
