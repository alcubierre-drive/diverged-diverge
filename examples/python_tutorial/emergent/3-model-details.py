#!/usr/bin/env python3
#coding:utf8
import diverge
import numpy as np

# initialization, needed for MPI compiled library
diverge.mpi_init( None, None )

# some parameters
params_emery = dict( Ud=3.0, Up=1.0, tdd=0.2, tpp=0.3, tdp=1.0 )
params_frg = dict( nk=16, nkf=5, model_output="emery_model.dvg" )

# we now use the actual model structure
model = diverge.model_init()
model.contents.name = b'Emery'
model.contents.n_orb = 3
model.contents.SU2 = True
model.contents.n_spin = 1
model.contents.nk[0] = params_frg['nk']
model.contents.nk[1] = params_frg['nk']
model.contents.nkf[0] = params_frg['nkf']
model.contents.nkf[1] = params_frg['nkf']

# lattice and positions
lattice = diverge.view_array( model.contents.lattice, dtype=np.float64, shape=(3,3) )
lattice[:,:] = np.eye(3)
positions = diverge.view_array( model.contents.positions, dtype=np.float64, shape=(3,3) )
positions[:,:] = np.array( [[0,0,0], [0.5,0,0], [0,0.5,0]] )

# get the hopping phase between p orbitals
def get_phase( o1, o2, po1, po2 ):
    if o1 == o2 and o1 == 0:
        return 1.0
    elif o1 == 0 and o2 == 1:
        return -1.0 if po2[0] > 0 else 1.0
    elif o1 == 0 and o2 == 2:
        return 1.0 if po2[1] > 0 else -1.0
    elif o1 == 1 and o2 == 0:
        dist = po1-po2
        return -1.0 if dist[0] > 0 else 1.0
    elif o1 == 2 and o2 == 0:
        dist = po1-po2
        return 1.0 if dist[1] > 0 else -1.0
    else:
        dist = po2-po1
        if dist[0] > 0:
            return -1.0 if dist[1] > 0 else 1.0
        else:
            return -1.0 if dist[1] < 0 else 1.0

# hopping parameters
hop_list = []
tdd_norm = 1.0
tdp_norm = 0.5
tpp_norm = 0.5*np.sqrt(2.)
for Rx in range(-2,3):
    for Ry in range(-2,3):
        for o1 in range(3):
            for o2 in range(3):
                dist = np.linalg.norm(lattice[0]*Rx + lattice[1]*Ry + positions[o2] - positions[o1])
                phase = get_phase(o1,o2,positions[o1],lattice[0]*Rx + lattice[1]*Ry + positions[o2])
                if (np.isclose(dist, tdd_norm) and o1 == 0 and o2 == 0):
                    hop_list.append( ( [Rx, Ry, 0], o1, o2, 0,0, (-params_emery['tdd']*phase,0.0) ) )
                if (np.isclose(dist, tpp_norm) and o1 != 0 and o2 != 0):
                    hop_list.append( ( [Rx, Ry, 0], o1, o2, 0,0, (-params_emery['tpp']*phase,0.0) ) )
                if (np.isclose(dist, tdp_norm)):
                    hop_list.append( ( [Rx, Ry, 0], o1, o2, 0,0, (-params_emery['tdp']*phase,0.0) ) )
hop_array = diverge.alloc_array( (len(hop_list),), dtype="rs_hopping_t" )
hop_array[:] = hop_list
diverge.mpi_py_eprint( "using %i hopping elements" % len(hop_list) )
# set the hopping pointer
model.contents.hop = hop_array.ctypes.data
model.contents.n_hop = hop_array.size

# vertices
vertices = diverge.alloc_array( (3,), dtype="rs_vertex_t" )
vertices[0] = ( 'D', [0,0,0], 0,0, -1,0,0,0, (params_emery['Ud'],0.0) )
vertices[1] = ( 'D', [0,0,0], 1,1, -1,0,0,0, (params_emery['Up'],0.0) )
vertices[2] = ( 'D', [0,0,0], 2,2, -1,0,0,0, (params_emery['Up'],0.0) )
# set the vertex pointer
model.contents.vert = vertices.ctypes.data
model.contents.n_vert = vertices.size

# set the irreducible path
ibz_path = diverge.view_array( model.contents.ibz_path, dtype=np.float64, shape=(4,3) )
model.contents.n_ibz_path = 4
ibz_path[:,:] = np.array( [ [0,0,0], [0,0.5,0], [0.5,0.5,0], [0,0,0] ] )

# validate the model
if diverge.model_validate( model ):
    diverge.mpi_py_eprint( "invalid model!" )
    exit(1)

# and initialize internal structures
diverge.model_internals_common( model )

# write to file
checksum = diverge.model_to_file( model, params_frg['model_output'], True )
diverge.mpi_py_eprint( "wrote model to file %s (%s)" % (params_frg['model_output'], checksum) )

# free resources
diverge.model_free( model )

# we need to accept that MPI requires finalization
diverge.mpi_finalize()
