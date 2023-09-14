#!/usr/bin/env python3
#coding:utf8
import diverge
import numpy as np

diverge.init(None, None)
diverge.compilation_status()
model = diverge.model_init()

model.contents.name = b'spinless_square'
model.contents.n_orb = 2
L = diverge.view_array( model.contents.lattice, dtype=np.float64, shape=(3,3) )
L[:,:] = np.eye(3)
L[0,:2] = [1, 1]
L[1,:2] = [1, -1]
positions = diverge.view_array( model.contents.positions, dtype=np.float64, shape=(2,3) )
positions[0,:2] = [0,0]
positions[1,:2] = [1,0]

model.contents.SU2 = False
model.contents.n_spin = 1
model.contents.nk[0] = 16
model.contents.nk[1] = 16
model.contents.nkf[0] = 1
model.contents.nkf[1] = 1
mu = 0.2
t = 1
tp = 0.2
D = 10

nn_norm = 1.0
nnn_norm = np.sqrt(2.)
hoppings = diverge.alloc_array( (1024,), "rs_hopping_t" )
nhop = 0
for Rx in range(-2,3):
    for Ry in range(-2,3):
        for o1 in range(2):
            for o2 in range(2):
                dist = np.linalg.norm(L[0]*Rx + L[1]*Ry - positions[o1] + positions[o2])
                if np.isclose(dist, nn_norm):
                    hoppings[nhop] = ( [Rx, Ry, 0], o1, o2, 0,0, (t,0.0) )
                    nhop = nhop+1
                if np.isclose(dist, nnn_norm):
                    hoppings[nhop] = ( [Rx, Ry, 0], o1, o2, 0,0, (-tp,0.0) )
                    nhop = nhop+1
hoppings[nhop] = ( [0, 0, 0], 0, 0, 0,0, (D,0.0) )
nhop = nhop+1
model.contents.hop = hoppings.ctypes.data
model.contents.n_hop = nhop

V = 2
vertex = diverge.alloc_array( (1024,), dtype="rs_vertex_t" )
nvert = 0
for Rx in range(-2,3):
    for Ry in range(-2,3):
        for o1 in range(2):
            for o2 in range(2):
                dist = np.linalg.norm(L[0]*Rx + L[1]*Ry - positions[o1] + positions[o2])
                if np.isclose(dist, nn_norm):
                    vertex[nvert] = ('D', [Rx, Ry, 0], o1, o2, 0,0,0,0, (V,0.0) )
                    nvert = nvert+1
model.contents.vert = vertex.ctypes.data
model.contents.n_vert = nvert

# we want a band structure
ibz_path = diverge.view_array( model.contents.ibz_path, dtype=np.float64, shape=(4,3) )
ibz_path[1,0] = 0.5
ibz_path[2,:2] = [0.5, 0.5]
model.contents.n_ibz_path = ibz_path.shape[0]

# validate model and initialize internals
if diverge.model_validate( model ):
    diverge.mpi_py_eprint("invalid model")
diverge.model_internals_common( model )
diverge.model_internals_tu( model, 3.1 )


checksum = diverge.model_to_file( model, "model.dvg" )
diverge.model_set_chempot( model, None, -1, mu )

step = diverge.flow_step_init( model, "tu", "PCDS" )
eu = diverge.euler_defaults_CPP()
maxs = dict(vert=np.zeros(1), loop=np.zeros(2), chan=np.zeros(3))

eu.Lambda = 10.0
eu.dLambda = -0.1
eu.dLambda_min = 0.1
eu.maxvert = 1e5
eu.maxiter = 10

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

diverge.postprocess_and_write( step, "out.dvg" )
diverge.flow_step_free( step )
diverge.model_free( model )
diverge.finalize()
