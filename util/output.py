#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math

# Namespace: dvg-output
# python library to read diverge output files

DIVERGE_MODEL_MAGIC_NUMBER = ord('D') + ord('M')
DIVERGE_POST_GRID_MAGIC_NUMBER = ord('P')
DIVERGE_POST_PATCH_MAGIC_NUMBER = ord('V')
DIVERGE_POST_TU_MAGIC_NUMBER = ord('T')

# return 1 if A>B, -1 if A<B, and 0 if A=B
def version_compare( A, B ):
    # special cases
    if (A == B):
        return 0
    if (A == "dev"):
        return 1
    if (B == "dev"):
        return -1

    # equalize lengths
    A_list = [int(x) for x in A.lstrip('v').split('.')]
    B_list = [int(x) for x in B.lstrip('v').split('.')]
    vlen = max( len(A_list), len(B_list) )
    if (len(A_list) < vlen):
        A_list += [0] * (vlen - len(A_list))
    if (len(B_list) < vlen):
        B_list += [0] * (vlen - len(B_list))

    # and check for larger version
    for i in range(vlen):
        if A_list[i] > B_list[i]:
            return 1
        elif A_list[i] < B_list[i]:
            return -1
        else:
            pass

    # default: versions are equal
    return 0

# this is due to some weird numpy version checking for the contents of a buffer
def numpy_fromfile( fname, **kwargs ):
    return np.frombuffer( open(fname, "rb").read(), **kwargs )

def qmatrices( data, displ, count, nq ):
    data_d = data[displ:displ+count]
    data_l = data_d.view( dtype=np.longlong )
    def advance( x ):
        nonlocal data_l
        nonlocal data_d
        data_l = data_l[x:]
        data_d = data_d[x:]
    result = []
    for i in range(nq):
        n_k_kp = data_l[0]
        advance(1)
        nb = data_l[0]
        advance(1)
        q = data_l[0]
        advance(1)
        eigen = data_l[0]
        advance(1)
        nv = data_l[0]
        advance(1)
        k_kp_descr = data_l[:n_k_kp].copy()
        advance(n_k_kp)
        if nv == -1 or eigen == 0:
            matrix_d = data_d[:n_k_kp*n_k_kp*nb*nb*nb*nb*2].copy()
            advance(n_k_kp*n_k_kp*nb*nb*nb*nb*2)
            matrix = (matrix_d[:-1:2] + 1j * matrix_d[1::2]).reshape((n_k_kp,nb,nb,n_k_kp,nb,nb))
            values = None
        elif eigen == 1 and nv > 0:
            values_d = data_d[:nv].copy()
            matrix_d = data_d[nv:nv+n_k_kp*nb*nb*2*nv].copy()
            advance(n_k_kp*nb*nb*2*nv + nv)
            matrix = (matrix_d[:-1:2] + 1j * matrix_d[1::2]).reshape((nv,n_k_kp,nb,nb))
            values = values_d
        else:
            print("ERROR. invalid parameters found for packed q-vertex")
        # TODO put eigen/nv info there
        res = (q, k_kp_descr, values, matrix)
        result.append( res )
    return result

def k_ibz_path( byte_ary_slice ):
    if byte_ary_slice.size == 0:
        return None
    else:
        i64 = np.int64
        i8 = i64().itemsize
        n_segments = byte_ary_slice[:i8].view(dtype=i64)[0]
        n_per_segment = byte_ary_slice[i8*1 : i8*(1+n_segments)].view(dtype=i64)
        n_path = byte_ary_slice[i8*(1+n_segments):i8*(2+n_segments)].view(dtype=i64)[0]
        path = byte_ary_slice[i8*(2+n_segments):i8*(2+n_segments+n_path)].view(dtype=i64)
        rest = byte_ary_slice[i8*(2+n_segments+n_path):].view(dtype=np.float64)
        return (n_per_segment, path, rest.reshape((-1,3)))

# Class: diverge_model
# documentation found in docstring, i.e., do the following from within python
# === Python ===
# help(diverge_model)
# ==============
class diverge_model:
    """class to read diverge model files

    name: model name
    version: diverge tag version (serves as file format version)
    dim: dimension
    nk[3]: #kpts
    nkf[3]: #fine kpts

    n_patches: #patches for npatch
    patches: patch indices wrt coarse kmesh
    weights: weight of each patch in BZ integral
    p_count: #refined points per patch
    p_displ: displacement of refined point array for each patch
    p_map: refined indices, indexed by p_count/p_displ
    p_weights: weights of refined indices, indexed by p_count/p_displ

    kc_ibz_path: IBZ path on the coarse momentum mesh as tuple (n_per_segment, path_indices, path_vectors)
    kf_ibz_path: IBZ path on the fine momentum mesh as tuple (n_per_segment, path_indices, path_vectors)

    ibz_path: IBZ path in crystal coordinates

    n_orb: #orbitals
    n_spin: #spins
    SU2: SU2 symmetry?
    lattice: lattice vectors (3,3)
    rlattice: reciprocal lattice vectors (3,3)
    positions: atom positions (n_orb, 3)

    n_sym: #symmetries
    orb_symmetries: orbital symmetries (n_sym, n_orb*n_spin, n_orb*n_spin)
    rs_symmetries: realspace symmetries (n_sym, 3, 3)

    n_hop: #hoppings
    hop: hopping elements (n_hop, custom struct)

    n_vert: #vertex elements
    vert: vertex elements (n_vert, custom struct)

    len_ff_struct: #formfactors
    tu_ff: formfactors (len_ff_struct, custom struct)

    n_vert_chan: #vertex elements in each channel (3)
    data: additional data (bytes)

    kmesh: coarse kmesh (nk[0]*nk[1]*nk[2], 3)
    kfmesh: fine kmesh (nk[0]*nk[1]*nk[2]*nkf[0]*nkf[1]*nkf[2], 3)
    ham: hamiltonain on fine kmesh (kfmesh.shape[0], n_spin*n_orb, n_spin*n_orb)
    U: eigenvectors on fine kmesh (kfmesh.shape[0], n_spin*n_orb, n_spin*n_orb)
    E: eigenvalues on fine kmesh (kfmesh.shape[0], n_spin*n_orb)

    npath: configuration for bandstructure (nonzero after v0.4.x) if -1, used
           kf_ibz_path indices. otherwise, bandstructure array is shaped as
           below
    bandstructure: banstructure along irreducible path, with irreducible path
                   in last three indices. shape usually given as
                   ((ibz_path.size-1)*300+1, n_spin*n_orb + 3) if not specified
                   differently in model output call.
    fatbands: if supplied (using npath == -1), contains array of shape
              (#kf_ibz_path, no_ns, nb) fatbands (nonzero after v0.5.x_dev)
    """

    def _displ_count( self, i_d ):
        return self._f_bytes[self._f_header[i_d]:self._f_header[i_d]+self._f_header[i_d+1]]
    def _head( self, i ):
        return self._f_header[i]
    def _head_displ_count( self, d, c ):
        return self._f_header[d:d+c]

    def __init__(self, fname):
        self._f_header = numpy_fromfile(fname, dtype=np.int64, count=128)

        def check_if_model():
            return (self._f_header[0] == DIVERGE_MODEL_MAGIC_NUMBER)

        def check_numerical_repr():
            checkmask = np.unpackbits(self._f_header[127:].view( dtype=np.uint8 ))
            if checkmask.sum() == 0:
                return True
            else:
                error_bits = np.where(checkmask)[0]
                print( "found errors in bits", error_bits, "(see diverge_model_output.h)" )
                return False

        if not check_if_model():
            self.valid = False
            return
        else:
            self.valid = True

        if not check_numerical_repr():
            self.valid = False

        self.rs_hopping_t = np.dtype(dict(
            names=['R', 'o1', 'o2', 's1', 's2', 't'],
            formats=['3i8', 'i8', 'i8', 'i8', 'i8', 'c16'],
            offsets=self._head_displ_count(102, 6),
            itemsize=self._head(108)))
        self.rs_vertex_t = np.dtype(dict(
            names=['chan', 'R', 'o1', 'o2', 's1', 's2', 's3', 's4', 'V'],
            formats=['b', '3i8', 'i8', 'i8', 'i8', 'i8', 'i8', 'i8', 'c16'],
            offsets=self._head_displ_count(109, 9),
            itemsize=self._head(118)))
        self.tu_formfactor_t = np.dtype(dict(
            names=['R', 'ofrom', 'oto', 'd', 'ffidx'],
            formats=['3i8', 'i8', 'i8', 'f8', 'i8'],
            offsets=self._head_displ_count(119, 5),
            itemsize=self._head(124)))

        self._f_bytes = numpy_fromfile(fname, dtype=np.byte)

        self.name = self._displ_count( 1 ).tobytes().decode().rstrip('\x00')
        self.version = self._displ_count( 125 ).tobytes().decode()
        self.dim = self._head(3)
        self.nk = self._head_displ_count(4,3)
        self.nkf = self._head_displ_count(7,3)

        self.n_patches = self._head(10)
        self.patches = self._displ_count( 11 ).view( dtype=np.int64 )
        self.weights = self._displ_count( 13 ).view( dtype=np.float64 )
        self.p_count = self._displ_count( 15 ).view( dtype=np.int64 )
        self.p_displ = self._displ_count( 17 ).view( dtype=np.int64 )
        self.p_map = self._displ_count( 19 ).view( dtype=np.int64 )
        self.p_weights = self._displ_count( 21 ).view( dtype=np.float64 )

        self.ibz_path = self._displ_count( 23 ).view( dtype=np.float64 ).reshape((-1,3))

        self.n_orb = self._head(25)
        self.n_spin = self._head(46)
        self.SU2 = self._head(45)
        self.lattice = self._head_displ_count(26, 9).view( dtype=np.float64 ).reshape((3,3))
        self.rlattice = self._head_displ_count(58, 9).view( dtype=np.float64 ).reshape((3,3))
        self.positions = self._displ_count( 35 ).view( dtype=np.float64 ).reshape((-1,3))

        self.n_sym = self._head(37)
        self.orb_symmetries = self._displ_count( 38 ).view( dtype=np.complex128 ).reshape(
                (-1, self.n_orb*self.n_spin, self.n_orb*self.n_spin))
        self.rs_symmetries = self._displ_count( 40 ).view( dtype=np.float64 ).reshape((-1,3,3))

        self.n_hop = self._head(42)
        self.hop = self._displ_count( 43 ).view( self.rs_hopping_t )

        self.n_vert = self._head(47)
        self.vert = self._displ_count( 48 ).view( self.rs_vertex_t )

        self.len_ff_struct = self._head( 50 )
        self.tu_ff = self._displ_count( 51 ).view( self.tu_formfactor_t )

        self.n_vert_chan = self._head_displ_count( 53, 3 )
        self.data = self._displ_count( 56 )

        self.kmesh = self._displ_count( 80 ).view( np.float64 ).reshape((-1,3))
        self.kfmesh = self._displ_count( 82 ).view( np.float64 ).reshape((-1,3))
        self.ham = self._displ_count( 84 ).view( np.complex128 ).reshape(
                (-1,self.n_spin*self.n_orb,self.n_spin*self.n_orb))
        self.U = self._displ_count( 86 ).view( np.complex128 ).reshape(
                (-1,self.n_spin*self.n_orb,self.n_spin*self.n_orb))
        self.E = self._displ_count( 88 ).view( np.float64 ).reshape((-1,self.n_spin*self.n_orb))

        self.kc_ibz_path = k_ibz_path( self._displ_count( 90 ) )
        self.kf_ibz_path = k_ibz_path( self._displ_count( 92 ) )

        self.npath = self._head(94) # to see what the config actually is
        self.bandstructure = self._displ_count( 100 ).view(
                np.float64 ).reshape((-1,self.n_spin*self.n_orb + 3))

        fatbands = self._displ_count( 98 ).view( np.float64 )
        if fatbands.size == 0:
            self.fatbands = None
        else:
            self.fatbands = fatbands.reshape( (-1, self.n_spin*self.n_orb, self.n_spin*self.n_orb) )

# Class: diverge_post_patch
# documentation found in docstring, i.e., do the following from within python
# === Python ===
# help(diverge_post_patch)
# ==============
class diverge_post_patch:
    '''class to read diverge postprocessing files for $N$-patch FRG

    version: diverge tag version (serves as file format version)

    vertices and loops may be None if not explicitly asked for in the
    postprocessing routines. They all are of shape (np, np, np, nb, nb, nb, nb),
    i.e. refer to the patch indices.

    V: full vertex
    Lp: particle particle loop
    Lm: particle hole loop
    dV: increment of the last step

    for each interaction channel X (X \in \{P,C,D\}), there are the following
    variables:

    X_chan: boolean that tracks whether this channel is included in the output
    X_dV: boolean that tracks whether the increment was used (True) or whether
          the vertex at scale was used (False)
    X_nq: the number of momentum transfer points
    X_qV: the vertx in 'q representation'. A list of length X_nq with each
          element of the list a tuple (q, k_kp_descr, values, vectors).
          q: transfer momentum
          k_kp_descr: array of indices of secondary momenta. Refers to the
                      coarse kmesh. May differ for different q.
          values: eigenvalues (shape: (nv,) with nv the number of eigenvalues
                  requested) may be None if no eigendecomposition was done
          vectors: eigenvectors (shape: (nv, len(k_kp_descr), nb, nb)). if
                   values is None, nv == len(k_kp_descr)*nb*nb and the array
                   should represents the vertex at the specified q as matrix
                   with shape: (len(k_kp_descr), nb, nb, len(k_kp_descr), nb, nb).
    '''
    def __init__(self, fname):
        d_header = numpy_fromfile(fname, dtype=np.longlong, count=128)

        if d_header[0] == DIVERGE_POST_PATCH_MAGIC_NUMBER:
            self.valid = True
        else:
            self.valid = False
            return

        d_data = numpy_fromfile(fname)
        d_data_l = numpy_fromfile(fname, dtype=np.longlong)

        if d_header[2] != 0:
            self.V = d_data[ 0+d_header[1] : d_header[1]+d_header[2] : 2 ] \
               + 1j* d_data[ 1+d_header[1] : d_header[1]+d_header[2] : 2 ]
        else:
            self.V = None
        if d_header[4] != 0:
            self.Lp = d_data[ 0+d_header[3] : d_header[3]+d_header[4] : 2 ] \
                + 1j* d_data[ 1+d_header[3] : d_header[3]+d_header[4] : 2 ]
        else:
            self.Lp = None
        if d_header[6] != 0:
            self.Lm = d_data[ 0+d_header[5] : d_header[5]+d_header[6] : 2 ] \
                + 1j* d_data[ 1+d_header[5] : d_header[5]+d_header[6] : 2 ]
        else:
            self.Lm = None
        if d_header[8] != 0:
            self.dV = d_data[ 0+d_header[7] : d_header[7]+d_header[8] : 2 ] \
                + 1j* d_data[ 1+d_header[7] : d_header[7]+d_header[8] : 2 ]
        else:
            self.dV = None

        idx = 9
        self.P_chan = chr(d_header[idx]); idx = idx+1
        self.P_dV = bool(d_header[idx]); idx = idx+1
        self.P_nq = d_header[idx]; idx = idx+1
        if d_header[idx+1] > 0:
            self.P_qV = qmatrices( d_data, d_header[idx], d_header[idx+1], self.P_nq ); idx = idx+2
        else:
            self.P_qV = None

        idx = 14
        self.C_chan = chr(d_header[idx]); idx = idx+1
        self.C_dV = bool(d_header[idx]); idx = idx+1
        self.C_nq = d_header[idx]; idx = idx+1
        if d_header[idx+1] > 0:
            self.C_qV = qmatrices( d_data, d_header[idx], d_header[idx+1], self.C_nq ); idx = idx+2
        else:
            self.C_qV = None

        idx = 19
        self.D_chan = chr(d_header[idx]); idx = idx+1
        self.D_dV = bool(d_header[idx]); idx = idx+1
        self.D_nq = d_header[idx]; idx = idx+1
        if d_header[idx+1] > 0:
            self.D_qV = qmatrices( d_data, d_header[idx], d_header[idx+1], self.D_nq ); idx = idx+2
        else:
            self.D_qV = None

        self.version = d_data[ d_header[125] : d_header[125]+d_header[126] ].view(
                dtype=np.byte ).tobytes().decode().rstrip('\x00')

# Class: diverge_post_grid
# documentation found in docstring, i.e., do the following from within python
# === Python ===
# help(diverge_post_grid)
# ==============
class diverge_post_grid:
    '''class to read diverge postprocessing files for grid FRG

    version: diverge tag version (serves as file format version)

    nk: number of k points
    nb: number of bands
    nff: number of formfactors
    SU2: SU2 symmetry
    lingap_num_ev: linearized gap equation number of singular values
    lingap_matrix_size: linearized gap equation matrix size

    formfac: formfactors (nff,nk)

    P_susc: P susceptibility (nk,nff,nb,nb,nb,nb)
    P_mf_U: P lingap U matrix (lingap_num_ev,lingap_matrix_size)
    P_mf_V: P lingap V matrix (lingap_num_ev,lingap_matrix_size)
    P_mf_S: P lingap singular values (lingap_num_ev)
    P_mf_EU: P lingap vertex eigenvectors (lingap_num_ev,lingap_matrix_size)
    P_mf_EV: P lingap vertex eigenvalues (lingap_num_ev)
    C_susc: C susceptibility (nk,nff,nb,nb,nb,nb)
    C_mf_U: C lingap U matrix (lingap_num_ev,lingap_matrix_size)
    C_mf_V: C lingap V matrix (lingap_num_ev,lingap_matrix_size)
    C_mf_S: C lingap singular values (lingap_num_ev)
    C_mf_EU: C lingap vertex eigenvectors (lingap_num_ev,lingap_matrix_size)
    C_mf_EV: C lingap vertex eigenvalues (lingap_num_ev)
    D_susc: D susceptibility (nk,nff,nb,nb,nb,nb)
    D_mf_U: D lingap U matrix (lingap_num_ev,lingap_matrix_size)
    D_mf_V: D lingap V matrix (lingap_num_ev,lingap_matrix_size)
    D_mf_S: D lingap singular values (lingap_num_ev)
    D_mf_EU: D lingap vertex eigenvectors (lingap_num_ev,lingap_matrix_size)
    D_mf_EV: D lingap vertex eigenvalues (lingap_num_ev)

    any of the above can be 'None' if it was not contained in simulation
    '''
    def __init__(self, fname):
        d_header = numpy_fromfile(fname, dtype=np.longlong, count=64)

        if d_header[0] == DIVERGE_POST_GRID_MAGIC_NUMBER:
            self.valid = True
        else:
            self.valid = False
            return

        d_data = numpy_fromfile(fname)

        self.version = d_data[d_header[60]:d_header[60]+d_header[61]].view(
                dtype=np.byte ).tobytes().decode().rstrip('\x00')

        self.nk = d_header[1]
        self.nb = d_header[2]
        self.nff = d_header[3]
        self.SU2 = d_header[4]

        self.lingap_num_ev = d_header[19]
        self.lingap_matrix_size = d_header[20]

        self.file_size = d_header[63]

        formfac = d_data[d_header[5]:d_header[5]+d_header[6]].reshape((self.nff,self.nk,2))
        self.formfac = formfac[:,:,0] + 1j * formfac[:,:,1]

        susc_shape = (self.nk,self.nff,self.nb,self.nb,self.nb,self.nb,2)

        _P_susc = d_data[d_header[7]:d_header[7]+d_header[8]]
        if _P_susc.size != 0:
            self.P_susc = _P_susc.reshape(susc_shape)
            self.P_susc = self.P_susc[:,:, :,:,:,:, 0] + 1j * self.P_susc[:,:, :,:,:,:, 1]
        else:
            self.P_susc = None
        _P_mf = d_data[d_header[9]:d_header[9]+d_header[10]]
        if _P_mf.size != 0:
            self.P_mf_U, self.P_mf_V, self.P_mf_S, self.P_mf_EU, self.P_mf_EV = self.unroll_mf_solution( _P_mf )
        else:
            self.P_mf_U = None
            self.P_mf_V = None
            self.P_mf_S = None
            self.P_mf_EU = None
            self.P_mf_EV = None

        _C_susc = d_data[d_header[11]:d_header[11]+d_header[12]]
        if _C_susc.size != 0:
            self.C_susc = _C_susc.reshape(susc_shape)
            self.C_susc = self.C_susc[:,:, :,:,:,:, 0] + 1j * self.C_susc[:,:, :,:,:,:, 1]
        else:
            self.C_susc = None
        _C_mf = d_data[d_header[13]:d_header[13]+d_header[14]]
        if _C_mf.size != 0:
            self.C_mf_U, self.C_mf_V, self.C_mf_S, self.C_mf_EU, self.C_mf_EV = self.unroll_mf_solution( _C_mf )
        else:
            self.C_mf_U = None
            self.C_mf_V = None
            self.C_mf_S = None
            self.C_mf_EU = None
            self.C_mf_EV = None

        _D_susc = d_data[d_header[15]:d_header[15]+d_header[16]]
        if _D_susc.size != 0:
            self.D_susc = _D_susc.reshape(susc_shape)
            self.D_susc = self.D_susc[:,:, :,:,:,:, 0] + 1j * self.D_susc[:,:, :,:,:,:, 1]
        else:
            self.D_susc = None
        _D_mf = d_data[d_header[17]:d_header[17]+d_header[18]]
        if _D_mf.size != 0:
            self.D_mf_U, self.D_mf_V, self.D_mf_S, self.D_mf_EU, self.D_mf_EV = self.unroll_mf_solution( _D_mf )
        else:
            self.D_mf_U = None
            self.D_mf_V = None
            self.D_mf_S = None
            self.D_mf_EU = None
            self.D_mf_EV = None

    def unroll_mf_solution( self, mf_solution ):
        U = mf_solution[:self.lingap_matrix_size * self.lingap_num_ev * 2]
        V = mf_solution[self.lingap_matrix_size * self.lingap_num_ev * 2:self.lingap_matrix_size * self.lingap_num_ev * 4]
        S = mf_solution[self.lingap_matrix_size * self.lingap_num_ev * 4:self.lingap_matrix_size * self.lingap_num_ev * 4 + self.lingap_num_ev]

        # only present due to change in code (now eigenvalues of vertex are
        # included...)
        if mf_solution.size > self.lingap_matrix_size * self.lingap_num_ev * 4 + self.lingap_num_ev:
            EU = mf_solution[self.lingap_matrix_size * self.lingap_num_ev * 4+self.lingap_num_ev: self.lingap_matrix_size*self.lingap_num_ev*6 +   self.lingap_num_ev]
            EV = mf_solution[self.lingap_matrix_size * self.lingap_num_ev * 6+self.lingap_num_ev: self.lingap_matrix_size*self.lingap_num_ev*6 + 2*self.lingap_num_ev]
        else:
            EU = None
            EV = None

        UU = U.reshape((self.lingap_num_ev,self.lingap_matrix_size,2))
        VV = V.reshape((self.lingap_num_ev,self.lingap_matrix_size,2))
        SS = S.reshape((self.lingap_num_ev,))

        if not (EU is None or EV is None):
            EU = EU.reshape((self.lingap_num_ev,self.lingap_matrix_size,2))
            EU = EU[:,:,0] + 1j * EU[:,:,1]
            EV = EV.reshape((self.lingap_num_ev,))
        return UU[:,:,0] + 1j * UU[:,:,1], VV[:,:,0] + 1j * VV[:,:,1], S, EU, EV

# Class: diverge_post_tu
# documentation found in docstring, i.e., do the following from within python
# === Python ===
# help(diverge_post_tu)
# ==============
class diverge_post_tu:
    '''class to read diverge postprocessing files for tu FRG

    version: diverge tag version (serves as file format version)

    n_orb: number of orbitals (not spin)
    n_spin: number of spin degrees of freedom
    nk: number of momentum points
    nktot: number of momentum for Hamiltonian
    nkibz: number of momentum points in the irreducible BZ wedge
    SU2: Exploit SU2 symmetry
    n_orbff: number of orbital+bond combinations
    n_bonds: maximal number of bonds of a single site, not necessarily the same 
            for every site
    mi_to_ofrom: map from n_orbff to (o,b) notation, stores o (n_orbff)
    mi_to_oto: map from n_orbff to (o,b) notation, stores o+bo (n_orbff)
    mi_to_R: map from n_orbff to (o,b) notation, stores the beyond unit cell
            contributiom (n_orbff,3)
    bond_sizes: number of bonds per site o, needed for correct iteration over
            tu_ff stored in model (n_orb)
    bond_offsets: offset to the bonds belonging to site o (n_orb)
    idx_ibz_in_full_mesh: gives the index of the IBZ point in the full PZ (nkibz)

    kmaps_to: symmetry mapping of full BZ to IBZ (nk)
    mi_to_tuffidx: multiindex to tu_ff index mapping; useful for analysis with model.tu_ff (n_orbff)

    VERTEX DIAGONALISATION
    Plen: for each q point the number of stored elements (nkibz)
    Poff: for each q point the offset in the array (nkibz)
    Ptype: for each q point the used diagonalization algorithm (nkibz)
    Pval: stored Eigenvalues (sum(Plen))
    Pvec: stored Eigenvectors (sum(Plen), n_orb*n_bonds*n_spin**2)

    Clen: for each q point the number of stored elements (nkibz)
    Coff: for each q point the offset in the array (nkibz)
    Ctype: for each q point the used diagonalization algorithm (nkibz)
    Cval: stored Eigenvalues (sum(Clen))
    Cvec: stored Eigenvectors (sum(Clen), n_orb*n_bonds*n_spin**2)

    Dlen: for each q point the number of stored elements (nkibz)
    Doff: for each q point the offset in the array (nkibz)
    Dtype: for each q point the used diagonalization algorithm (nkibz)
    Dval: stored Eigenvalues (sum(Dlen))
    Dvec: stored Eigenvectors (sum(Dlen), n_orb*n_bonds*n_spin**2)

    LINGAP solution
    S_sc: singular values SC lingap (n_sing_val)
    U_sc: U SC lingap (n_sing_val, n_orbff*n_spin**2)
    V_sc: V SC lingap (n_sing_val, n_orbff*n_spin**2)

    S_mag: singular values magnetic lingap (n_sing_val)
    U_mag: U magnetic lingap (n_sing_val, n_orbff*n_spin**2)
    V_mag: V magnetic lingap (n_sing_val, n_orbff*n_spin**2)

    S_ch: singular values charge lingap (n_sing_val)
    U_ch: U charge (n_sing_val, n_orbff*n_spin**2)
    V_ch: V charge (n_sing_val, n_orbff*n_spin**2)

    SUSCEPTIBILITY
    Psusc: Pair-Pair susceptibility ([n_orb*n_spin]*4,nkibz)
    Csusc: magnetic/crossed PH susceptibility ([n_orb*n_spin]*4,nkibz)
    Dsusc: charge/direct PH susceptibility ([n_orb*n_spin]*4,nkibz)
    Psuscff: formfactor resolved Pair-Pair susceptibility ([n_orbff*n_spin*n_spin]*2,nkibz)
    Csuscff: formfactor resolved magnetic/crossed PH susceptibility ([n_orbff*n_spin*n_spin]*2,nkibz)
    Dsuscff: formfactor resolved charge/direct PH susceptibility (n_orbff*n_spin*n_spin]*2,nkibz)
    any of the above can be 'None' if it was not contained in simulation

    SELFENERGY
    selfenergy: Self-energy at the critical scale ([n_orb*n_spin]*2,nktot)


    FULL channels
    Pchannel: P channel on full PZ
    Cchannel: C channel on full PZ
    Dchannel: D channel on full PZ
    pploop: non interacting PP-loop channel on full PZ
    phloop: non interacting PH-loop channel on full PZ

    SYMMETRIES
    symm_o2m_len: length of symmetry maps
    symm_o2m_off: offsets of symmetry maps
    symm_o2m_idx_map: index maps
    self.symm_o2m_Ppref: complex prefactor for P
    self.symm_o2m_Cpref: complex prefactor for C
    '''
    def __init__(self, fname):
        d_header = numpy_fromfile(fname, dtype=np.int64, count=128)

        if d_header[0] == DIVERGE_POST_TU_MAGIC_NUMBER and d_header[126] == DIVERGE_POST_TU_MAGIC_NUMBER:
            self.valid = True
        else:
            self.valid = False
            return

        self.n_orb = int(d_header[1])
        self.n_spin = int(d_header[2])
        self.nk = int(d_header[3])
        self.nktot = int(d_header[4])
        self.nkibz = int(d_header[5])
        self.SU2 = int(d_header[6])
        self.n_orbff = int(d_header[7])
        self.n_bonds = int(d_header[8])
        self.n_sym = int(d_header[9])
        n_spin = self.n_spin
        n_bonds = self.n_bonds
        n_orb = self.n_orb
        n_orbff = self.n_orbff

        _f_bytes = numpy_fromfile(fname, dtype=np.byte)
        def get_array( offset_bytes, count, dtype ):
            return _f_bytes[offset_bytes:offset_bytes + count*dtype().itemsize].view( dtype=dtype )
        def get_array_header( iL, dtype ):
            res = get_array( d_header[iL[0]], d_header[iL[0]+1], dtype )
            iL[0] = iL[0] + 2
            return res

        iL = [10] # can't pass by reference, therefore we need a list... python is stupid after all
        self.mi_to_ofrom = get_array_header( iL, np.int64 )
        self.mi_to_oto = get_array_header( iL, np.longlong )
        self.mi_to_R = get_array_header( iL, np.longlong ).reshape((-1,3))
        self.bond_sizes = get_array_header( iL, np.longlong )
        self.bond_offsets = get_array_header( iL, np.longlong )
        self.idx_ibz_in_bz = get_array_header( iL, np.longlong )
        self.Plen = get_array_header( iL, np.longlong )
        self.Poff = get_array_header( iL, np.longlong )
        self.Ptype = get_array_header( iL, np.byte )
        self.Pval = get_array_header( iL, np.float64 )
        self.Pvec = get_array_header( iL, np.complex128 )

        self.Pvec = self.Pvec.reshape((np.sum(self.Plen),n_spin,n_spin,n_orbff))

        self.Clen = get_array_header( iL, np.longlong )
        self.Coff = get_array_header( iL, np.longlong )
        self.Ctype = get_array_header( iL, np.byte )
        self.Cval = get_array_header( iL, np.float64 )
        self.Cvec = get_array_header( iL, np.complex128 )

        self.Cvec = self.Cvec.reshape((np.sum(self.Clen),n_spin,n_spin,n_orbff))

        self.Dlen = get_array_header( iL, np.longlong )
        self.Doff = get_array_header( iL, np.longlong )
        self.Dtype = get_array_header( iL, np.byte )
        self.Dval = get_array_header( iL, np.float64 )
        self.Dvec = get_array_header( iL, np.complex128 )

        self.Dvec = self.Dvec.reshape((np.sum(self.Dlen),n_spin,n_spin,n_orbff))

        self.S_sc = get_array_header( iL, np.float64 )
        self.n_svdP = d_header[iL[0]-1]
        self.U_sc = get_array_header( iL, np.complex128 )
        self.U_sc = self.U_sc.reshape((self.n_svdP,n_spin,n_spin,n_orbff))
        self.V_sc = get_array_header( iL, np.complex128 )
        self.V_sc = self.V_sc.reshape((self.n_svdP,n_spin,n_spin,n_orbff))

        self.S_mag = get_array_header( iL, np.float64 )
        self.n_svdC = d_header[iL[0]-1]
        self.U_mag = get_array_header( iL, np.complex128 )
        self.U_mag = self.U_mag.reshape((self.n_svdC,n_spin,n_spin,n_orbff))
        self.V_mag = get_array_header( iL, np.complex128 )
        self.V_mag = self.V_mag.reshape((self.n_svdC,n_spin,n_spin,n_orbff))

        self.S_charge = get_array_header( iL, np.float64 )
        self.n_svdD = d_header[iL[0]-1]
        self.U_charge = get_array_header( iL, np.complex128 )
        self.U_charge = self.U_charge.reshape((self.n_svdD,n_spin,n_spin,n_orbff))
        self.V_charge = get_array_header( iL, np.complex128 )
        self.V_charge = self.V_charge.reshape((self.n_svdD,n_spin,n_spin,n_orbff))

        susc_shape = (n_spin,n_orb,n_spin,n_orb,n_spin,n_orb,n_spin,n_orb,self.nk)
        self.Psusc = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Psusc = self.Psusc.reshape(susc_shape)
        self.Csusc = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Csusc = self.Csusc.reshape(susc_shape)
        self.Dsusc = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Dsusc = self.Dsusc.reshape(susc_shape)

        susc_shape = (n_spin*n_spin,n_orbff,n_spin*n_spin,n_orbff,self.nk)
        self.Psusc_ff = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Psusc_ff = self.Psusc_ff.reshape(susc_shape)
        self.Csusc_ff = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Csusc_ff = self.Csusc_ff.reshape(susc_shape)
        self.Dsusc_ff = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Dsusc_ff = self.Dsusc_ff.reshape(susc_shape)

        self.selfenergy = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.selfenergy = self.selfenergy.reshape(self.nktot,n_spin,n_orb,n_spin,n_orb)

        self.Pchannel = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Pchannel = self.Pchannel.reshape(susc_shape)
        self.Cchannel = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Cchannel = self.Cchannel.reshape(susc_shape)
        self.Dchannel = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Dchannel = self.Dchannel.reshape(susc_shape)
        self.pploop = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.pploop = self.pploop.reshape(susc_shape)
        self.phloop = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.phloop = self.phloop.reshape(susc_shape)

        symm_shape = (self.n_sym,n_spin,n_spin,n_orbff)
        self.symm_o2m_len = get_array_header( iL, np.longlong )
        if(d_header[iL[0]-1] > 0):
            self.symm_o2m_len = self.symm_o2m_len.reshape(symm_shape)
        self.symm_o2m_off = get_array_header( iL, np.longlong )
        if(d_header[iL[0]-1] > 0):
            self.symm_o2m_off = self.symm_o2m_off.reshape(symm_shape)
        self.symm_o2m_idx_map = get_array_header( iL, np.longlong )
        self.symm_o2m_Ppref = get_array_header( iL, np.complex128 )
        self.symm_o2m_Cpref = get_array_header( iL, np.complex128 )

        self.version = get_array_header( iL, np.byte ).tobytes().decode()

        self.kmaps_to = get_array_header( iL, np.longlong )
        self.mi_to_tuffidx = get_array_header( iL, np.longlong )

# Function: diverge_read
# function to read model files as well as post processing files. returns the
# corresponding class (<diverge_model>, <diverge_post_patch>, or
# <diverge_post_grid>). Makes use of to the file format specifications given in
# <diverge_postprocess_and_write at
# diverge_postprocess_conf_t.diverge_postprocess_and_write> and
# <diverge_model_to_file>; discerning the different file types by their magic
# numbers
#
# Parameters:
# fname - file name to read from (typically ....dvg)
def read( fname ):
    magic = numpy_fromfile( fname, dtype='i8', count=1 )[0]
    if magic == DIVERGE_MODEL_MAGIC_NUMBER:
        return diverge_model( fname )
    elif magic == DIVERGE_POST_PATCH_MAGIC_NUMBER:
        return diverge_post_patch( fname )
    elif magic == DIVERGE_POST_GRID_MAGIC_NUMBER:
        return diverge_post_grid( fname )
    elif magic == DIVERGE_POST_TU_MAGIC_NUMBER:
        return diverge_post_tu( fname )
    else:
        print(f"magic number {magic} ('{chr(magic)}') unknown")
        return None

# Function: bandstructure_bands
# return the bands array (nk,nb) from the band  structure obtained from a
# <diverge_model>. Useful for plotting (for example see <bandstructure_xvals>).
def bandstructure_bands( model ):
    return model.bandstructure[:,:-3]

# Function: bandstructure_kpts
# return the momentum points (nk,3) from the band structure obtained from a
# <diverge_model>. Useful for plotting and post-processing.
def bandstructure_kpts( model ):
    return model.bandstructure[:,-3:]

# Function: bandstructure_xvals
# calculate the differential distance between the momentum points from the band
# structure obtained from a <diverge_model>. Useful for plotting in conjuction
# with <bandstructure_bands> and <bandstructure_ticks>.
# === Python ===
# import diverge.output as do
# import matplotlib.pyplot as plt
# M = do.read("model.dvg")
# b = do.bandstructure_bands(M)
# x = do.bandstructure_xvals(M)
# plt.plot( x, b, color='black' )
# plt.xticks( do.bandstructure_ticks(M) )
# # need to set the labels manually as they are not known to the model
# plt.show()
# ==============
def bandstructure_xvals( model ):
    K = bandstructure_kpts(model)
    vals = np.concatenate( [[0], np.sqrt((np.diff(K,axis=0)**2).sum(axis=1))] )
    if model.npath == -1 and not model.kf_ibz_path is None:
        n_per_segment = model.kf_ibz_path[0]
        offsets = np.cumsum( n_per_segment )
        offset_zero = offsets[n_per_segment == 0]
        vals[offset_zero] = 0.0
    return np.cumsum(vals)

# Function: bandstructure_ticks
# returns the xticks used for band structure plots from a <diverge_model>.
# Useful in conjuction with <bandstructure_xvals>.
def bandstructure_ticks( model ):
    x = bandstructure_xvals( model )
    n_bandstruct = x.shape[0]
    n_ibz_path = model.ibz_path.shape[0]

    if n_ibz_path <= 1:
        return None

    ticks = []

    if model.npath == 0: # we don't know what the call was. trying to guess.
        n_per_tick_float = (n_bandstruct - 1)/(n_ibz_path-1)
        n_per_tick_int = (n_bandstruct - 1)//(n_ibz_path-1)
        if math.isclose(n_per_tick_int, n_per_tick_float):
            ticks += list( x[ np.arange(n_ibz_path) * n_per_tick_int ] )
        else:
            if not model.kf_ibz_path is None:
                n_per_segment = model.kf_ibz_path[0]
                ticks += list( x[np.concatenate( [[0], np.cumsum( n_per_segment ) - 1] )] )

    elif model.npath >= 1:
        ticks += list( x[ np.arange(n_ibz_path) * model.npath ] )
    elif model.npath == -1:
        if not model.kf_ibz_path is None:
            n_per_segment = model.kf_ibz_path[0]
            ticks += list( x[np.concatenate( [[0], np.cumsum( n_per_segment ) - 1] )] )
        else:
            print( "missing kf_ibz_path in model file" )

    if len(ticks) == n_ibz_path:
        return ticks
    else:
        return None



if __name__ == "__main__":
    # mod = read('post.dvg')
    # print(mod.Plen)
    '''
    print("model name:", mod.name)
    import matplotlib.pyplot as plt
    plt.scatter( mod.kmesh[:,0], mod.kmesh[:,1], c=mod.E[:,0], rasterized=True, lw=0, s=1, cmap=plt.cm.gray_r )
    plt.scatter( *mod.kmesh[mod.patches,:2].T, color='k', s=10 )
    def k1p2( k1, k2, nkx, nky ):
        return ((k1 // nky + k2 // nky) % (nkx)) * nky + (k1 % nky + k2 % nky) % (nky)
    for p in range(len(mod.patches)):
        c = mod.p_count[p]
        d = mod.p_displ[p]
        plt.scatter(*mod.kmesh[k1p2(mod.patches[p], mod.p_map[d:d+c], mod.nk[0], mod.nk[1]),:2].T, lw=0, s=2)
    plt.gca().set_aspect('equal')
    plt.show()
    '''
