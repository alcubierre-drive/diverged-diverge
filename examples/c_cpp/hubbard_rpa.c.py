#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import fancymaps as fm

n_kpts = 1074
n_vals = 10

d = np.fromfile('hubbard_rpa.c.bin', dtype=np.int64)
kpts = d[:n_kpts]
vals = d[n_kpts:n_kpts+n_vals].view(dtype=np.float64)
U = d[n_kpts+n_vals:n_kpts+n_vals+n_vals*n_kpts*2].view(dtype=np.complex128).reshape((n_vals,n_kpts))
V = d[n_kpts+n_vals+n_vals*n_kpts*2:].view(dtype=np.complex128).reshape((n_vals,n_kpts))

kx = kpts//128
ky = kpts%128

ccmap = fm.complex_cmap_factory(fm.Cycl_wdiv4)
plt.figure()
plt.scatter( kx, ky, c=ccmap(U[0]), s=2 )
plt.title( 'U' )
plt.figure()
plt.scatter( kx, ky, c=ccmap(V[0]), s=2 )
plt.title( 'V' )
plt.show()
