#!/usr/bin/env python3
#coding:utf8

import diverge
import numpy as np
import sys

def t2g_model( U = 1.0, J=0.1, nk=24, nkf=1,
                 tu_maxdist=2.01, outfile="mod.dvg" ):
    model = diverge.model_init()
    model.contents.name = b't2g_1d'
    model.contents.n_orb = 3
    model.contents.SU2 = True
    model.contents.n_spin = 1
    model.contents.nk[0] = nk
    model.contents.nkf[0] = nkf

    L = diverge.view_array( model.contents.lattice, dtype=np.float64, shape=(3,3) )
    L[:,:] = np.eye(3)


    hoppings = diverge.alloc_array( (6,), "rs_hopping_t" )
    nhop = 6
    hoppings[0] = ([1,0,0],0,0,0,0,1.2)
    hoppings[1] = ([-1,0,0],0,0,0,0,1.2)
    hoppings[2] = ([1,0,0],1,1,0,0,1)
    hoppings[3] = ([-1,0,0],1,1,0,0,1)
    hoppings[4] = ([1,0,0],2,2,0,0,1)
    hoppings[5] = ([-1,0,0],2,2,0,0,1)
    model.contents.hop = hoppings.ctypes.data
    model.contents.n_hop = nhop

    vertex = diverge.alloc_array( (128,), dtype="rs_vertex_t" )
    elem = 0;
    for o1 in range(3):
        for o2 in range(3):
            if(o1 == o2):
                vertex[elem] = ('D', [0,0,0],o1,o2,0,0,0,0,(U,0.0))
                elem+=1
            else:
                vertex[elem] = ('D', [0,0,0],o1,o2,0,0,0,0,(U-2.*J,0.0))
                elem+=1
                vertex[elem] = ('C', [0,0,0],o1,o2,0,0,0,0,(J,0.0))
                elem+=1
                vertex[elem] = ('P', [0,0,0],o1,o2,0,0,0,0,(J,0.0))
                elem+=1
    model.contents.vert = vertex.ctypes.data
    model.contents.n_vert = vertex.size

    syms = np.zeros( 1, dtype=diverge.sym_op_t )
    orbs = np.zeros( 3, dtype=diverge.site_descr_t )
    orbs[0]['amplitude'][0] = (1.0, 0.0)
    orbs[0]['function'][0] = diverge.orb_dxy
    orbs[0]['n_functions'] = 1
    orbs[1]['amplitude'][0] = (1.0, 0.0)
    orbs[1]['function'][0] = diverge.orb_dxz
    orbs[1]['n_functions'] = 1
    orbs[2]['amplitude'][0] = (1.0, 0.0)
    orbs[2]['function'][0] = diverge.orb_dyz
    orbs[2]['n_functions'] = 1
    nsym = 2
    model.contents.n_sym = nsym
    orb_symmetries = diverge.alloc_array( (nsym,3,3), dtype="complex128_t" )
    rs_symmetries = diverge.view_array( model.contents.rs_symmetries, dtype=np.float64, shape=(nsym,3,3) )
    syms[0]["normal_vector"] = [0,0,1.0]
    syms[0]["type"] = "E"
    ### Generate symmetries
    rs_symmetries[0], orb_symmetries[0] = diverge.generate_symm_trafo(1,orbs,syms)
    syms[0]["type"] = "R"
    syms[0]["angle"] = 180
    rs_symmetries[1], orb_symmetries[1] = diverge.generate_symm_trafo(1,orbs,syms)
    model.contents.orb_symmetries = orb_symmetries.ctypes.data
    # validate model and initialize internals
    if diverge.model_validate( model ):
        diverge.mpi_py_eprint("invalid model")
    model.contents.n_sym *= -1
    diverge.model_internals_common( model )
    diverge.model_internals_tu( model, tu_maxdist )

    checksum = diverge.model_to_file( model, outfile )
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
    diverge.postprocess_and_write( step, outfile )

def cleanup( model, step ):
    diverge.flow_step_free( step )
    diverge.model_free( model )

diverge.init(None, None)
diverge.compilation_status()

model = t2g_model( nk=24, nkf = 1, outfile="t2g_model.py_mod.dvg")
step = flow( model, outfile="t2g_model.py_flow.dat" )
post( step, outfile="t2g_model.py_out.dvg" )
cleanup( model, step )
diverge.finalize()
