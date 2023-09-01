#!/usr/bin/env python3
#coding:utf8

import diverge as dvg
import numpy as np

dvg.mpi_init( None, None )
dvg.compilation_status()

# initialize
model = dvg.model_init()

# name
model.contents.name = b'Nickelate'

# kpts
model.contents.nk[0] = 8
model.contents.nk[1] = 8
model.contents.nk[2] = 8
model.contents.nkf[0] = 5
model.contents.nkf[1] = 5
model.contents.nkf[2] = 5

# lattice (tetragonal)
lattice = dvg.view_array( model.contents.lattice, dtype=np.float64, shape=(3,3) )
lattice[0,0] = 3.92
lattice[1,1] = 3.92
lattice[2,2] = 3.28

# orbitals
model.contents.n_orb = 3

# and sites
positions = dvg.view_array( model.contents.positions, dtype=np.float64, shape=(3,3) )
# Ni: origin, defaults to zero!
# Re: 1/2, 1/2, 1/2
positions[1,:] = np.array([1.96,1.96,1.64])
positions[2,:] = np.array([1.96,1.96,1.64])

# read hoppings from W90 file
hoppings_w90 = dvg.read_W90_PY( "nickelate_hr.dat", 0 ) # nspin = 0 for SU2 symmetry

# save+allocate them in the model
hop = dvg.alloc_array( (hoppings_w90.size,), dtype="rs_hopping_t" )
hop[:] = hoppings_w90
model.contents.hop = hop.ctypes.data
model.contents.n_hop = hoppings_w90.size

# SU2 symmetry
model.contents.SU2 = True
model.contents.n_spin = 1

model.contents.n_ibz_path = 8;
ibz_path = dvg.view_array( model.contents.ibz_path, dtype=np.float64, shape=(8,3) )
# Gamma
ibz_path[1,:] = np.array( [0.5,0.5,0] ) # M
ibz_path[2,:] = np.array( [0.5,0,0] ) # X
# Gamma
ibz_path[4,:] = np.array( [0,0,0.5] ) # Z
ibz_path[5,:] = np.array( [0.5,0.5,0.5] ) # A
ibz_path[6,:] = np.array( [0.5,0,0.5] ) # R
ibz_path[7,:] = np.array( [0,0,0.5] ) # Z

# and finally, we want to validate!
if dvg.model_validate( model ):
    dvg.mpi_exit(1)

# make the code set its internals
dvg.model_internals_common( model )
dvg.model_internals_tu( model, 3.92 * 1.1 )

# and output the model
dvg.model_to_file( model, "nickelate_model.dvg", True ) # common internals

# and free
dvg.model_free( model )

dvg.mpi_finalize()
