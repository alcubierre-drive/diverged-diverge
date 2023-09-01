#!/usr/bin/env python3
#coding:utf8

import diverge
import numpy as np
import sys

def get_phase(o1,o2,po1,po2):
    if(o1 == o2 and o1 == 0):
        return 1.0
    elif((o1 == 0 and o2 == 1)):
        if(po2[0] > 0):
            return -1.
        else:
            return 1.
    elif((o1 == 0 and o2 == 2)):
        if(po2[1] > 0):
            return 1.
        else:
            return -1.
    elif((o1 == 1 and o2 == 0)):
        dist = po1-po2
        if(dist[0] > 0):
            return -1.
        else:
            return 1.
    elif((o1 == 2 and o2 == 0)):
        dist = po1-po2
        if(dist[1] > 0):
            return 1.
        else:
            return -1.
    else:
        dist = po2-po1
        if(dist[0] > 0):
            if(dist[1] > 0):
                return -1.
            else:
                return 1.
        else:
            if(dist[1] < 0):
                return -1.
            else:
                return 1.


def emery_model( Ud=3.0, Up=1.0, tdd=0.1,tpp=0.3, tdp=1.0,
                nk=24, nkf=1, tu_maxdist=2.01,
                outfile="mod.dvg" ):

    # basic setup
    model = diverge.model_init()
    model.contents.name = b'Emery'
    model.contents.n_orb = 3
    model.contents.SU2 = True
    model.contents.n_spin = 1
    model.contents.nk[0] = nk
    model.contents.nk[1] = nk
    model.contents.nkf[0] = nkf
    model.contents.nkf[1] = nkf

    # lattice
    L = diverge.view_array( model.contents.lattice, dtype=np.float64, shape=(3,3) )
    L[:,:] = np.eye(3)
    L[0,:2] = [1.0,0.0]
    L[1,:2] = [0.0,1.0]

    # positions
    positions = diverge.view_array( model.contents.positions, dtype=np.float64, shape=(3,3) )
    positions[1,:] = [0.5,0.0,0.0]
    positions[2,:] = [0.0,0.5,0.0]

    tdd_norm = 1.0
    tdp_norm = 0.5
    tpp_norm = 0.5*np.sqrt(2.)
    hoppings = diverge.alloc_array( (1024,), "rs_hopping_t" )
    nhop = 0

    # hopping
    for Rx in range(-2,3):
        for Ry in range(-2,3):
            for o1 in range(3):
                for o2 in range(3):
                    dist = np.linalg.norm(L[0]*Rx + L[1]*Ry + positions[o2] - positions[o1])
                    phase = get_phase(o1,o2,positions[o1],L[0]*Rx + L[1]*Ry + positions[o2])
                    if (np.isclose(dist, tdd_norm) and o1 == 0 and o2 == 0):
                        hoppings[nhop] = ( [Rx, Ry, 0], o1, o2, 0,0, (-tdd*phase,0.0) )
                        nhop = nhop+1
                    if (np.isclose(dist, tpp_norm) and o1 != 0 and o2 != 0):
                        hoppings[nhop] = ( [Rx, Ry, 0], o1, o2, 0,0, (-tpp*phase,0.0) )
                        nhop = nhop+1
                    if (np.isclose(dist, tdp_norm)):
                        hoppings[nhop] = ( [Rx, Ry, 0], o1, o2, 0,0, (-tdp*phase,0.0) )
                        nhop = nhop+1
    model.contents.hop = hoppings.ctypes.data
    model.contents.n_hop = nhop

    # vertex
    vertex = diverge.alloc_array( (3,), dtype="rs_vertex_t" )
    vertex[0] = ('D', [0,0,0], 0,0, 0,0,0,0, (Ud,0.0))
    vertex[1] = ('D', [0,0,0], 1,1, 0,0,0,0, (Up,0.0))
    vertex[2] = ('D', [0,0,0], 2,2, 0,0,0,0, (Up,0.0))
    model.contents.vert = vertex.ctypes.data
    model.contents.n_vert = vertex.size

    ### set ibz_path
    ibz_path = diverge.view_array( model.contents.ibz_path, dtype=np.float64, shape=(4,3) )
    ibz_path[1,:] = [0.5,0,0]
    ibz_path[2,:] = [0.5,0.5,0]
    model.contents.n_ibz_path = ibz_path.shape[0]

    ### set orbital
    syms = np.zeros( 1, dtype=diverge.sym_op_t )
    orbs = np.zeros( 3, dtype=diverge.site_descr_t )
    orbs[0]['amplitude'][0] = (1.0, 0.0)
    orbs[0]['function'][0] = diverge.orb_dx2y2
    orbs[0]['n_functions'] = 1
    orbs[1]['amplitude'][0] = (1.0, 0.0)
    orbs[1]['function'][0] = diverge.orb_px
    orbs[1]['n_functions'] = 1
    orbs[2]['amplitude'][0] = (1.0, 0.0)
    orbs[2]['function'][0] = diverge.orb_py
    orbs[2]['n_functions'] = 1

    ### Allocate symmetry buffer
    nsym = 8
    model.contents.n_sym = nsym

    orb_symmetries = diverge.alloc_array( (nsym,3,3), dtype="complex128_t" )
    model.contents.orb_symmetries = orb_symmetries.ctypes.data
    rs_symmetries = diverge.view_array( model.contents.rs_symmetries, dtype=np.float64, shape=(nsym,3,3) )
    syms[0]["normal_vector"] = [0,0,1.0]
    syms[0]["type"] = "E"
    ### Generate symmetries
    rs_symmetries[0], orb_symmetries[0] = diverge.generate_symm_trafo(1,orbs,syms)
    syms[0]["type"] = "R"
    syms[0]["angle"] = 90
    rs_symmetries[1], orb_symmetries[1] = diverge.generate_symm_trafo(1,orbs,syms)
    syms[0]["angle"] = 2*90
    rs_symmetries[2], orb_symmetries[2] = diverge.generate_symm_trafo(1,orbs,syms)
    syms[0]["angle"] = 3*90
    rs_symmetries[3], orb_symmetries[3] = diverge.generate_symm_trafo(1,orbs,syms)
    syms[0]["type"] = "M"
    syms[0]["normal_vector"] = [1.0,0,0.0]
    rs_symmetries[4], orb_symmetries[4] = diverge.generate_symm_trafo(1,orbs,syms)
    syms[0]["normal_vector"] = [0.0,1.0,0.0]
    rs_symmetries[5], orb_symmetries[5] = diverge.generate_symm_trafo(1,orbs,syms)
    syms[0]["normal_vector"] = [1.0,1.0,0.0]
    rs_symmetries[6], orb_symmetries[6] = diverge.generate_symm_trafo(1,orbs,syms)
    syms[0]["normal_vector"] = [1.0,-1.0,0.0]
    rs_symmetries[7], orb_symmetries[7] = diverge.generate_symm_trafo(1,orbs,syms)

    # validate model and initialize internals
    if diverge.model_validate( model ):
        diverge.mpi_py_eprint("invalid model")
    diverge.model_internals_common( model )
    diverge.model_internals_tu( model, tu_maxdist )
    print("Symmetry error is: ",diverge.symmetrize_2pt_fine(model,diverge.model_internals_get_H(model),None))

    checksum = diverge.model_to_file( model, outfile, True )
    diverge.mpi_py_eprint("wrote to file '%s' (%s)" % (outfile, checksum))
    return model


def flow( model, maxvert=200, outfile="flow.dat", chans="PCD" ):
    step = diverge.flow_step_init( model, "tu", chans )

    maxs = dict(vert=np.zeros(1), loop=np.zeros(2), chan=np.zeros(3))
    eu = diverge.euler_defaults_CPP()
    eu.dLambda_fac_scale = 1.0
    eu.dLambda_fac = 0.1
    eu.maxvert = maxvert

    stdout = sys.stdout
    sys.stdout = diverge.mpi_stdout_logger( outfile )
    diverge.mpi_py_eprint( "flow output to %s…" % outfile )

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

    sys.stdout = stdout
    return step

def post( step, outfile="out.dvg" ):
    diverge.postprocess_and_write( step, None, outfile )

def cleanup( model, step ):
    diverge.flow_step_free( step )
    diverge.model_free( model )


diverge.mpi_init(None, None)
diverge.compilation_status()

model = emery_model( nk=24, nkf = 1, outfile="mod.dvg")
step = flow( model, outfile="flow.dat" )
post( step, outfile="out.dvg" )
cleanup( model, step )
