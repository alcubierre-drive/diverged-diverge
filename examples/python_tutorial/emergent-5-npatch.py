#!/usr/bin/env python3
#coding:utf8
import diverge
import numpy as np

# initialization, needed for MPI compiled library
diverge.init( None, None )

# some parameters
params_emery = dict( Ud=3.0, Up=1.0, tdd=0.2, tpp=0.3, tdp=1.0 )
params_frg = dict( nk=80, nkf=1, filling=0.8, npatches_ibz=6, model_output="model_npatch.dvg" )

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

# to reduce work or to generate a patching, we should make use of C4v symmetry
def symmetry_setup( model ):
    # what orbitals are involved?
    orbs = np.zeros( (3,), dtype=diverge.site_descr_t )
    orbs[0]['amplitude'][0] = (1.0, 0.0)
    orbs[0]['function'][0] = diverge.orb_dx2y2
    orbs[0]['n_functions'] = 1
    orbs[1]['amplitude'][0] = (1.0, 0.0)
    orbs[1]['function'][0] = diverge.orb_px
    orbs[1]['n_functions'] = 1
    orbs[2]['amplitude'][0] = (1.0, 0.0)
    orbs[2]['function'][0] = diverge.orb_py
    orbs[2]['n_functions'] = 1

    # symmetry operator chain
    syms = np.zeros( (1,), dtype=diverge.sym_op_t )

    # and allocation/mapping of the model structures
    orb_symmetries = diverge.alloc_array( (8,3,3), dtype="complex128_t" )
    model.contents.orb_symmetries = orb_symmetries.ctypes.data
    rs_symmetries = diverge.view_array( model.contents.rs_symmetries, dtype=np.float64, shape=(8,3,3) )
    model.contents.n_sym = 8

    # E
    syms[0]["normal_vector"] = [0,0,1.0]
    syms[0]["type"] = "E"
    rs_symmetries[0], orb_symmetries[0] = diverge.generate_symm_trafo(1,orbs,syms)
    # 2C4
    syms[0]["type"] = "R"
    syms[0]["angle"] = 90
    rs_symmetries[1], orb_symmetries[1] = diverge.generate_symm_trafo(1,orbs,syms)
    syms[0]["angle"] = 3*90
    rs_symmetries[3], orb_symmetries[3] = diverge.generate_symm_trafo(1,orbs,syms)
    # C2
    syms[0]["angle"] = 2*90
    rs_symmetries[2], orb_symmetries[2] = diverge.generate_symm_trafo(1,orbs,syms)
    # 2sigma_v
    syms[0]["type"] = "M"
    syms[0]["normal_vector"] = [1.0,0,0.0]
    rs_symmetries[4], orb_symmetries[4] = diverge.generate_symm_trafo(1,orbs,syms)
    syms[0]["normal_vector"] = [0.0,1.0,0.0]
    rs_symmetries[5], orb_symmetries[5] = diverge.generate_symm_trafo(1,orbs,syms)
    # 2sigma_d
    syms[0]["normal_vector"] = [1.0,1.0,0.0]
    rs_symmetries[6], orb_symmetries[6] = diverge.generate_symm_trafo(1,orbs,syms)
    syms[0]["normal_vector"] = [1.0,-1.0,0.0]
    rs_symmetries[7], orb_symmetries[7] = diverge.generate_symm_trafo(1,orbs,syms)
symmetry_setup( model )

# validate the model
if diverge.model_validate( model ):
    diverge.mpi_py_eprint( "invalid model!" )
    exit(1)

# and initialize internal structures
diverge.model_internals_common( model )

# set the filling close to upper vHs
# diverge_model_set_filling( model, energies, nbands, nu )
mu = diverge.model_set_filling( model, None, -1, params_frg['filling'] )

# generate patching with 6 patches in the irreducible wedge of the BZ
diverge.model_internals_patch( model, params_frg['npatches_ibz'] )

# write to file
checksum = diverge.model_to_file( model, params_frg['model_output'], True )
diverge.mpi_py_eprint( "wrote model to file %s (%s)" % (params_frg['model_output'], checksum) )

# free resources
diverge.model_free( model )

# we need to accept that MPI requires finalization
diverge.finalize()
