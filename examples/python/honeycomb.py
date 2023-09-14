#!/usr/bin/env python3
#coding:utf8

import diverge
import numpy as np
import sys
import matplotlib.pyplot as plt

def honeycomb_model( U=3.0, t=1.0, tp=0.235, mu=1.5, nk=1200, nkf=1,
                    mode="patch", tu_maxdist=0.7, patch_np_ibz=6,
                    outfile="mod.dvg" ):
    # commons
    model = diverge.model_init()
    model.contents.name = b'honeycomb_p_wave'
    model.contents.n_orb = 2
    model.contents.SU2 = True
    model.contents.n_spin = 1
    model.contents.nk[0] = nk
    model.contents.nk[1] = nk
    model.contents.nkf[0] = nkf
    model.contents.nkf[1] = nkf

    # lattice
    L = diverge.view_array( model.contents.lattice, dtype=np.float64, shape=(3,3) )
    L[:,:] = np.eye(3)
    L[0,:2] = [np.sin(np.pi/3),-np.cos(np.pi/3)]
    L[1,:2] = [np.sin(np.pi/3),np.cos(np.pi/3)]

    # positions
    positions = diverge.view_array( model.contents.positions, dtype=np.float64, shape=(2,3) )
    positions[:,:] = np.array( [ -1./3.*(L[0]+L[1]), 1./3.*(L[0]+L[1]) ] )
    # positions[:,:] = np.array( [ 1./3.*(L[0]+L[1]), 2./3.*(L[0]+L[1]) ] )

    # hoppings
    nn_norm = np.linalg.norm( positions[0] )
    nnn_norm = 1.0
    hoppings = diverge.alloc_array( (100,), "rs_hopping_t" )
    nhop = 0
    for Rx in range(-2,3):
        for Ry in range(-2,3):
            for o1 in range(2):
                for o2 in range(2):
                    dist = np.linalg.norm(L[0]*Rx + L[1]*Ry + positions[o2] - positions[o1])
                    if np.isclose(dist, nn_norm):
                        hoppings[nhop] = ( [Rx, Ry, 0], o1, o2, 0,0, (-t,0.0) )
                        nhop = nhop+1
                    if np.isclose(dist, nnn_norm):
                        hoppings[nhop] = ( [Rx, Ry, 0], o1, o2, 0,0, (-tp,0.0) )
                        nhop = nhop+1
                    if np.isclose(dist, 0.0):
                        hoppings[nhop] = ( [Rx, Ry, 0], o1, o2, 0,0, (-mu,0.0) )
                        nhop = nhop+1
    model.contents.hop = hoppings.ctypes.data
    model.contents.n_hop = nhop

    # vertex
    vertex = diverge.alloc_array( (2,), dtype="rs_vertex_t" )
    vertex[0] = ('D', [0,0,0], 0,0, 0,0,0,0, (U,0.0))
    vertex[1] = ('D', [0,0,0], 1,1, 0,0,0,0, (U,0.0))
    model.contents.vert = vertex.ctypes.data
    model.contents.n_vert = vertex.size

    # symmetries
    nsym = 12
    model.contents.n_sym = nsym

    orb_symmetries = diverge.alloc_array( (12,2,2), dtype="complex128_t" )
    model.contents.orb_symmetries = orb_symmetries.ctypes.data
    orb_symmetries[:,:,:] = np.ones((2,2))[None,:,:]
    rs_symmetries = diverge.view_array( model.contents.rs_symmetries, dtype=np.float64, shape=(nsym,3,3) )

    rot = lambda theta: np.array( [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]] )
    rx = np.array( [[-1,0],[0,1]] )
    ry = np.array( [[1,0],[0,-1]] )
    for i in range(6):
        rs_symmetries[i] = np.eye(3)
        rs_symmetries[i,:2,:2] = rot(np.pi*2/6*i)
    for i in range(6,9):
        Rp = rot( np.pi/3*(i-6))
        Rm = rot(-np.pi/3*(i-6))
        rs_symmetries[i] = np.eye(3)
        rs_symmetries[i,:2,:2] = Rp @ rx @ Rm
    for i in range(9,12):
        Rp = rot( np.pi/3*(i-9))
        Rm = rot(-np.pi/3*(i-9))
        rs_symmetries[i] = np.eye(3)
        rs_symmetries[i,:2,:2] = Rp @ ry @ Rm

    orb_symmetries[:] *= 0.0
    for s in range(12):
        for o in range(2):
            po = rs_symmetries[s] @ positions[o]
            try:
             for p in range(2):
              for r1 in range(-3,4):
               for r2 in range(-3,4):
                pp = positions[p] + L[0] * r1 + L[1] * r2
                if np.linalg.norm(po - pp) < 1e-7: raise IndexError
            except IndexError:
                orb_symmetries[s,o,p] = 1.0


    # we want a band structure
    ibz_path = diverge.view_array( model.contents.ibz_path, dtype=np.float64, shape=(4,3) )
    ibz_path[1,0] = 0.5
    ibz_path[2,:2] = [2./3., 1./3.]
    model.contents.n_ibz_path = ibz_path.shape[0]

    # validate model and initialize internals
    if diverge.model_validate( model ):
        diverge.mpi_py_eprint("invalid model")
    diverge.model_internals_common( model )

    if mode == "tu":
        diverge.model_internals_tu( model, tu_maxdist )
    elif mode == "patch":
        pts = np.array( np.concatenate( [
            diverge.patching_find_fs_pts_PY( model, None, 2, patch_np_ibz, 100 ),
            [0] ] ), dtype=diverge.index_t )

        model.contents.patching = diverge.patching_from_indices( model, pts.ctypes.data, pts.size )
        diverge.patching_autofine( model, model.contents.patching, None, 2, 40, 0.1, 2.0, 0.5 )
        diverge.patching_symmetrize_refinement( model, model.contents.patching )
        diverge.model_internals_patch( model, -1 )
    elif mode == "grid":
        diverge.model_internals_grid( model )
    else:
        diverge.mpi_py_eprint("unsupported mode. use 'tu' or 'patch'")

    checksum = diverge.model_to_file( model, outfile )
    diverge.mpi_py_eprint("wrote to file '%s' (%s)" % (outfile, checksum))

    return model

def flow( model, mode="patch", maxvert=200, outfile="flow.dat", chans="PCD" ):
    step = diverge.flow_step_init( model, mode, chans )

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

test_run = True
test_plot = True
skip_sym = False

if test_run:
    prefix = ''
    model = honeycomb_model( tp=0.15, mu=0.93, U=3.0, nk=300, patch_np_ibz=3, nkf=1,
            outfile=prefix+"T.dvg", mode="patch" )
    if skip_sym:
        diverge.model_hack( model, "npatch_vertex_resym", "0" )
    step = flow( model, outfile=prefix+"F.dvg", mode="patch" )
    post( step, outfile=prefix+"O.dvg" )
    cleanup( model, step )

if test_plot:
    def complex_cmap(ary):
        a = np.copy(ary.flatten())
        a /= np.abs(a).max()
        c = plt.cm.hsv( (np.angle(a) + np.pi)/(2.*np.pi) )
        c[:,3] *= np.abs(a)
        c /= c.max()
        return c.reshape( (*ary.shape, 4) )

    import diverge.output as ro
    O = ro.read('O.dvg')
    M = ro.read('T.dvg')

    plt.plot( M.bandstructure[:,:-3], c='k' )

    PQ = O.P_qV[0]
    kpt = M.kmesh[PQ[1],:2]
    Ubo = M.U[PQ[1]]
    val = PQ[2]
    vec = PQ[3]

    # print(val[val<0])
    # raise SystemExit
    for i in range(8):
        fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
        V = np.einsum( 'kop, kbo, kcp -> kbc', vec[i], Ubo.conj(), Ubo )
        V = vec[i]
        colors = complex_cmap(V)
        for o in range(2):
            for p in range(2):
                axs[o,p].scatter( *kpt.T, c=colors[:,o,p] )
                axs[o,p].set_aspect('equal')
        fig.suptitle( str(i) + ' : ' + str(val[i]) )
    plt.show()

if test_run or test_plot:
    raise SystemExit

mu_shifts = np.array([0.01,0.03,0.05,0.07,0.09,0.13])
tp_vals = np.round(np.arange(0.17,0.25,0.005),3)
for i_tp in range(len(tp_vals)):
    for j_mu in range(len(mu_shifts)):
        mu = 3.0-6.0*tp_vals[i_tp]-mu_shifts[j_mu]

        prefix = "tp%02d_mu%02d_" % (i_tp, j_mu)
        model = honeycomb_model( nk=150, outfile=prefix+"mod.dvg", patch_np_ibz=3 )
        step = flow( model, outfile=prefix+"flow.dat" )
        post( step, outfile=prefix+"out.dvg" )
        cleanup( model, step )

diverge.finalize()

