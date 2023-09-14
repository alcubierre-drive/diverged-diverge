#!/usr/bin/env python3
#coding:utf8

import diverge as dvg
import numpy as np

dvg.init( None, None )
dvg.compilation_status()

model = dvg.model_init()
model.contents.name = b'Nickelate'
model.contents.nk[0] = 4
model.contents.nk[1] = 4
model.contents.nk[2] = 4
model.contents.nkf[0] = 5
model.contents.nkf[1] = 5
model.contents.nkf[2] = 5
lattice = dvg.view_array( model.contents.lattice, dtype=np.float64, shape=(3,3) )
lattice[0,0] = 3.92
lattice[1,1] = 3.92
lattice[2,2] = 3.28
model.contents.n_orb = 3
positions = dvg.view_array( model.contents.positions, dtype=np.float64, shape=(3,3) )
positions[1,:] = np.array([1.96,1.96,1.64])
positions[2,:] = np.array([1.96,1.96,1.64])
hoppings_w90 = dvg.read_W90_PY( "nickelate_hr.dat", 0 ) # nspin = 0 for SU2 symmetry
hop = dvg.alloc_array( (hoppings_w90.size,), dtype="rs_hopping_t" )
hop[:] = hoppings_w90
model.contents.hop = hop.ctypes.data
model.contents.n_hop = hoppings_w90.size
model.contents.SU2 = True
model.contents.n_spin = 1


# Hubbard-Kanamori Hamiltonian
vert = dvg.alloc_array( (1024,), dtype="rs_vertex_t" )
U_Ni = 3.0
U_R = 2.0
J_R = 0.36
vert[0] = ('D', (0,0,0), 0,0, -1,0,0,0, (U_Ni,0));
vert[1] = ('D', (0,0,0), 1,1, -1,0,0,0, (U_R,0));
vert[2] = ('D', (0,0,0), 2,2, -1,0,0,0, (U_R,0));
vert[3] = ('D', (0,0,0), 1,2, -1,0,0,0, (U_R - 2.0*J_R,0));
vert[4] = ('D', (0,0,0), 2,1, -1,0,0,0, (U_R - 2.0*J_R,0));
vert[5] = ('C', (0,0,0), 1,2, -1,0,0,0, (J_R,0));
vert[6] = ('C', (0,0,0), 2,1, -1,0,0,0, (J_R,0));
vert[7] = ('P', (0,0,0), 1,2, -1,0,0,0, (J_R,0));
vert[8] = ('P', (0,0,0), 2,1, -1,0,0,0, (J_R,0));
model.contents.vert = vert.ctypes.data
model.contents.n_vert = 9

if dvg.model_validate( model ):
    dvg.mpi_exit(1)

dvg.model_internals_common( model )
dvg.model_internals_tu( model, 3.92 * 1.1 )

# flow step
step = dvg.flow_step_init( model, "tu", "PCD" )
euler = dvg.euler_defaults_CPP()
vertmax = np.array([0.0])
loopmax = np.array([0.0,0.0])
chanmax = np.array([0.0,0.0,0.0])
dvg.print( "Lambda dLambda Lp Lm P C D Vmax" )

while dvg.euler_next( dvg.byref(euler), vertmax[0] ):
    dvg.flow_step_euler( step, euler.Lambda, euler.dLambda )
    dvg.flow_step_vertmax( step, vertmax.ctypes.data )
    dvg.flow_step_loopmax( step, loopmax.ctypes.data )
    dvg.flow_step_chanmax( step, chanmax.ctypes.data )
    dvg.print( '%.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e' % \
            (euler.Lambda, euler.dLambda, *loopmax, *chanmax, *vertmax) )

# post process
postconf = dvg.postprocess_config_default_CPP()
postconf.tu_storing_threshold = 0.8
postconf.tu_storing_relative = True
dvg.postprocess_and_write( step, dvg.byref(postconf), "nickelate_post.dvg" )

# clean up
dvg.flow_step_free( step )

dvg.model_free( model )
dvg.finalize()
