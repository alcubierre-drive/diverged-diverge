#!/usr/bin/env python3
#coding:utf8

import diverge
import numpy as np

diverge.init(None,None)

model = diverge.model_init()
# define momentum grid
model.contents.nk[0] = 24
model.contents.nk[1] = 24
model.contents.nkf[0] = 1
model.contents.nkf[1] = 1

# set the real space lattice vectors from QE lattice relaxation
lattice = diverge.view_array( model.contents.lattice, dtype=np.float64, shape=(3,3) )
lattice[0,0] = 4.025913292782465
lattice[0,1] = -2.324362123375175
lattice[1,1] = 4.648724246750350
lattice[2,2] = 50.6443932121

model.contents.n_orb = 2
pos = diverge.view_array( model.contents.positions, dtype=np.float64, shape=(2,3) )

pos[0,:] = 0.0 * lattice[0,:] + 0.0 * lattice[1,:] - 0.062503511 * lattice[2,:]
pos[1,:] = lattice[0,:]/3. + 2.0 * lattice[1,:]/3. - 0.062496000 * lattice[2,:]

# SU2 symmetric - default value 0 amounts to SU(2) symmetric model
# negative spin means it is the index which increases memory slowly
# positive spin means it is the index increasing the memory fast
nspin = 0
path_to_W90 = "Graphene_hr.dat"
hop = diverge.read_W90_PY(path_to_W90, nspin)
# we could do an autoflow with the hoppings generated from W90!

model_hop = diverge.alloc_array( (hop.size,), dtype="rs_hopping_t" )
model_hop[:] = hop

model.contents.hop = model_hop.ctypes.data
model.contents.n_hop = model_hop.size
model.contents.SU2 = True
model.contents.n_spin = 1

ibz_path = diverge.view_array(model.contents.ibz_path,dtype=np.float64, shape=(4,3))
ibz_path[1,:] = [1./3.,1./3.,0.0]
ibz_path[2,:] = [0.0,0.5,0.0]
ibz_path[3,:] = [0.0,0.0,0.0]
model.contents.n_ibz_path = ibz_path.shape[0]

diverge.model_internals_common( model )

npath = 2000
diverge.model_output_set_npath( npath )
checksum = diverge.model_to_file( model, "mod.dvg", True )

diverge.model_free( model )
diverge.finalize()

import diverge.output as rd
model_result = rd.read( "mod.dvg" )

import matplotlib.pyplot as plt

# plot bandstructure. the model_result.bandstructure array contains all bands
# (in the first nb indices of axis 1) and the respective coordinates in momentum
# space (in the last three indices of axis 1). we use the momentum coordinates
# to correctly scale the xaxis lengths. by default, divERGe calculates the
# bandstructure for 300 points in each segment of the irreducible path. In this
# case, the shape of the bandstructure array therefore is (901, 4): 300 + 300 +
# 300 + 1 points with 1 band and 3 momentum coordinates each
plt.figure( figsize=(4,3), layout="constrained" )
xvals = np.concatenate( [[0], np.cumsum(np.sqrt((np.diff(model_result.bandstructure[:,-3:],axis=0)**2).sum(axis=1)))] )
plt.plot( xvals, model_result.bandstructure[:,:2], c='k' )
plt.xticks( [xvals[0], xvals[npath], xvals[2*npath], xvals[3*npath]], [r'$\Gamma$', r'$K$', r'$M$', r'$\Gamma$'] )
plt.xlim( xvals[0], xvals[-1] )
plt.ylabel( r'$\epsilon({\bf k})$' )
plt.show()

