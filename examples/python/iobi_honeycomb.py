#!/usr/bin/env python3
#coding:utf8

import diverge
import numpy as np
import sys
import diverge.output as ro
import matplotlib.pyplot as plt

diverge.init(None, None)
diverge.mpi_loglevel_set(6)
diverge.compilation_status()
model = diverge.model_init()
model.contents.name = b'honeycomb_p_wave'
model.contents.n_orb = 2
model.contents.SU2 = False
model.contents.n_spin = 2
model.contents.nk[0] = 6
model.contents.nk[1] = 6
model.contents.nkf[0] = 1
model.contents.nkf[1] = 1

tp = 0
t = 1
mu = 0.2
L = diverge.view_array( model.contents.lattice, dtype=np.float64, shape=(3,3) )
L[:,:] = np.eye(3)
L[1,:2] = [np.cos(np.pi/3), np.sin(np.pi/3)]
positions = diverge.view_array( model.contents.positions, dtype=np.float64, shape=(2,3) )
positions[:,:] = np.array( [ -1./3.*(L[0]+L[1]), 1./3.*(L[0]+L[1]) ] )
nn_norm = np.linalg.norm( positions[1] )
nnn_norm = 1.0
hoppings = diverge.alloc_array( (1024,), "rs_hopping_t" )
nhop = 0
for Rx in range(-2,3):
    for Ry in range(-2,3):
        for o1 in range(2):
            for o2 in range(2):
                dist = np.linalg.norm(L[0]*Rx + L[1]*Ry + positions[o1] - positions[o2])
                for s in range(2):
                    if np.isclose(dist, nn_norm):
                        hoppings[nhop] = ( [Rx, Ry, 0], o1, o2, s,s, (t,0.0) )
                        nhop = nhop+1
                    if np.isclose(dist, nnn_norm):
                        hoppings[nhop] = ( [Rx, Ry, 0], o1, o2, s,s, (-tp,0.0) )
                        nhop = nhop+1

B = 1.0
m = 0.3

hoppings[nhop] = ( [0,0,0], 0,0, 0,0, (B,0.0) )
nhop = nhop+1
hoppings[nhop] = ( [0,0,0], 0,0, 1,1, (-B,0.0) )
nhop = nhop+1

hoppings[nhop] = ( [0,0,0], 1,1, 0,0, (B,0.0) )
nhop = nhop+1
hoppings[nhop] = ( [0,0,0], 1,1, 1,1, (-B,0.0) )
nhop = nhop+1

for s in range(2):
    hoppings[nhop] = ( [0,0,0], 1,1, s,s, (m,0.0) )
    nhop = nhop+1

model.contents.hop = hoppings.ctypes.data
model.contents.n_hop = nhop

# vertex
vertex = diverge.alloc_array( (2,), dtype="rs_vertex_t" )
U = 3.0
vertex[0] = ('D', [0,0,0], 0,0, -1,0,0,0, (U,0.0))
vertex[1] = ('D', [0,0,0], 1,1, -1,0,0,0, (U,0.0))
model.contents.vert = vertex.ctypes.data
model.contents.n_vert = vertex.size

# we want a band structure
ibz_path = diverge.view_array( model.contents.ibz_path, dtype=np.float64, shape=(4,3) )
ibz_path[1,0] = 0.5
ibz_path[2,:2] = [2./3., 1./3.]
model.contents.n_ibz_path = ibz_path.shape[0]

# validate model and initialize internals
if diverge.model_validate( model ):
    diverge.mpi_py_eprint("invalid model")
diverge.model_internals_common( model )
diverge.model_internals_tu( model, diverge.max_dist_iobi() )
checksum = diverge.model_to_file( model, "hiobi_m.dvg" )
diverge.model_set_chempot( model, None, -1, mu )

step = diverge.flow_step_init( model, "tu", "CD" )
maxs = dict(vert=np.zeros(1), loop=np.zeros(2), chan=np.zeros(3))

eu = diverge.euler_defaults_CPP()
eu.dLambda_fac_scale = 1e-9
eu.dLambda_fac = 1e-9

eu.Lambda = 10.0
eu.dLambda = -0.1
eu.dLambda_min = 0.1
eu.maxvert = 1e5
eu.maxiter = 90

diverge.print( "# Lambda dLambda Lp Lm dP dC dD V" )
while True:
    diverge.flow_step_euler( step, eu.Lambda, eu.dLambda )
    diverge.flow_step_vertmax( step, maxs['vert'].ctypes.data )
    diverge.flow_step_loopmax( step, maxs['loop'].ctypes.data )
    diverge.flow_step_chanmax( step, maxs['chan'].ctypes.data )
    diverge.print( "%.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e" %
              (eu.Lambda, eu.dLambda, *maxs['loop'], *maxs['chan'], *maxs['vert']) )
    eu_next = diverge.euler_next( diverge.byref(eu), maxs['vert'][0] )
    if not eu_next:
        break

diverge.flow_step_free( step )
diverge.model_free( model )
diverge.finalize()

