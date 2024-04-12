#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math

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

class diverge_model:
    """
    class to read diverge model files (.dvg, "magic number" = 145)

    :param fname: file name of a divERGe model file that should be read, usually
                  ``XXX.dvg``.
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

        #: name given to the model by the user through :c:member:`diverge_model_t.name`
        self.name = self._displ_count( 1 ).tobytes().decode().rstrip('\x00')

        #: diverge tag version (serves as file format version)
        self.version = self._displ_count( 125 ).tobytes().decode()
        #: dimension
        self.dim = self._head(3)
        #: number of kpts as (3,) array of :c:type:`index_t`
        self.nk = self._head_displ_count(4,3)
        #: number of fine kpts as (3,) array of :c:type:`index_t`
        self.nkf = self._head_displ_count(7,3)

        #: number of patches (in case npatch backend was initalized)
        self.n_patches = self._head(10)
        #: indices of the coarse mesh that are patch centers (n_patches,)
        self.patches = self._displ_count( 11 ).view( dtype=np.int64 )
        #: weights for each of the patches (n_patches,)
        self.weights = self._displ_count( 13 ).view( dtype=np.float64 )
        #: descriptor for the ``p_map`` and ``p_weights`` arrays, such that
        #: ``p_map[p_displ[p]:p_displ[p]+p_count[p]]`` is the slice in the array
        #: ``p_map`` (in python notation) that makes the refinement for patch ``p``
        self.p_count = self._displ_count( 15 ).view( dtype=np.int64 )
        #: descriptor for the ``p_map`` and ``p_weights`` arrays, same as above
        self.p_displ = self._displ_count( 17 ).view( dtype=np.int64 )

        #: refinement indices for patches; *relative convention*. Let ``p`` be a
        #: patch index, ``i`` a refinement index for ``p`` (i.e. between zero and
        #: ``p_count[p]``). Then, the refinement index ``kfi = p_map[p_displ[p]+i]``
        #: is the relative shift from the patch center ``patches[p]`` to the
        #: actual refinement point, i.e., ``kf = k1p2(kfi, patches[p])``.
        self.p_map = self._displ_count( 19 ).view( dtype=np.int64 )
        #: refinement weights for patches, same indexing convention as ``p_map``.
        self.p_weights = self._displ_count( 21 ).view( dtype=np.float64 )

        #: IBZ path in crystal coordinates
        self.ibz_path = self._displ_count( 23 ).view( dtype=np.float64 ).reshape((-1,3))

        #: number of orbitals
        self.n_orb = self._head(25)
        #: number of spins :math:`n_\mathrm{spin} = 2S+1`.
        self.n_spin = self._head(46)
        #: :math:`SU(2)` symmetry?
        self.SU2 = self._head(45)
        #: lattice vectors as (3,3) array
        self.lattice = self._head_displ_count(26, 9).view( dtype=np.float64 ).reshape((3,3))
        #: reciprocal lattice vectors as (3,3) array
        self.rlattice = self._head_displ_count(58, 9).view( dtype=np.float64 ).reshape((3,3))
        #: positions as (n_orb,3) array
        self.positions = self._displ_count( 35 ).view( dtype=np.float64 ).reshape((-1,3))

        #: number of symmetries
        self.n_sym = self._head(37)
        #: orbital symmetries (n_sym, n_orb*n_spin, n_orb*n_spin)
        self.orb_symmetries = self._displ_count( 38 ).view( dtype=np.complex128 ).reshape(
                (-1, self.n_orb*self.n_spin, self.n_orb*self.n_spin))
        #: realspace symmetries (n_sym, 3, 3)
        self.rs_symmetries = self._displ_count( 40 ).view( dtype=np.float64 ).reshape((-1,3,3))

        #: number of hoppings
        self.n_hop = self._head(42)
        #: hopping elements (n_hop, custom struct)
        self.hop = self._displ_count( 43 ).view( self.rs_hopping_t )

        #: number of vertex elements
        self.n_vert = self._head(47)
        #: vertex elements (n_vert, custom struct)
        self.vert = self._displ_count( 48 ).view( self.rs_vertex_t )

        #: number of TU formfactors
        self.len_ff_struct = self._head( 50 )
        #: formfactors (len_ff_struct, custom struct). *Note: usually not
        #: present without TU internals*.
        self.tu_ff = self._displ_count( 51 ).view( self.tu_formfactor_t )

        #: number of vertex elements in each channel (3)
        self.n_vert_chan = self._head_displ_count( 53, 3 )
        #: additional data (byte array)
        self.data = self._displ_count( 56 )

        #: coarse kmesh (nk[0]*nk[1]*nk[2], 3).
        #: *Note: only present if (i) common internals were set at the time of
        #: saving and (ii) the output config key was set to a non-default, non-zero
        #: value.*
        self.kmesh = self._displ_count( 80 ).view( np.float64 ).reshape((-1,3))
        #: fine kmesh (nk[0]*nk[1]*nk[2]*nkf[0]*nkf[1]*nkf[2], 3)
        #: *Note: only present if (i) common internals were set at the time of
        #: saving and (ii) the output config key was set to a non-default, non-zero
        #: value.*
        self.kfmesh = self._displ_count( 82 ).view( np.float64 ).reshape((-1,3))
        #: hamiltonain on fine kmesh (kfmesh.shape[0], n_spin*n_orb, n_spin*n_orb)
        #: *Note: only present if (i) common internals were set at the time of
        #: saving and (ii) the output config key was set to a non-default, non-zero
        #: value.*
        self.ham = self._displ_count( 84 ).view( np.complex128 ).reshape(
                (-1,self.n_spin*self.n_orb,self.n_spin*self.n_orb))
        #: eigenvectors on fine kmesh (kfmesh.shape[0], n_spin*n_orb, n_spin*n_orb)
        #: *Note: only present if (i) common internals were set at the time of
        #: saving and (ii) the output config key was set to a non-default, non-zero
        #: value.*
        self.U = self._displ_count( 86 ).view( np.complex128 ).reshape(
                (-1,self.n_spin*self.n_orb,self.n_spin*self.n_orb))
        #: eigenvalues on fine kmesh (kfmesh.shape[0], n_spin*n_orb)
        #: *Note: only present if (i) common internals were set at the time of
        #: saving and (ii) the output config key was set to a non-default, non-zero
        #: value.*
        self.E = self._displ_count( 88 ).view( np.float64 ).reshape((-1,self.n_spin*self.n_orb))

        #: IBZ path on the coarse momentum mesh as tuple (n_per_segment, path_indices, path_vectors)
        self.kc_ibz_path = k_ibz_path( self._displ_count( 90 ) )
        #: IBZ path on the fine momentum mesh as tuple (n_per_segment, path_indices, path_vectors)
        self.kf_ibz_path = k_ibz_path( self._displ_count( 92 ) )

        #: Configuration for bandstructure (nonzero after v0.4.x). If -1, used
        #: kf_ibz_path indices. Otherwise, bandstructure array is shaped as below.
        self.npath = self._head(94) # to see what the config actually is

        #: banstructure along irreducible path, with irreducible path
        #: in last three indices. shape usually given as
        #: ((ibz_path.size-1)*300+1, n_spin*n_orb + 3) if not specified
        #: differently in model output call.
        self.bandstructure = self._displ_count( 100 ).view(
                np.float64 ).reshape((-1,self.n_spin*self.n_orb + 3))

        #: If supplied (using npath == -1 and the corresponding config key),
        #: contains array of shape (#kf_ibz_path, no_ns, nb) fatbands (nonzero
        #: after v0.6)
        fatbands = self._displ_count( 98 ).view( np.float64 )
        if fatbands.size == 0:
            self.fatbands = None
        else:
            self.fatbands = fatbands.reshape( (-1, self.n_spin*self.n_orb, self.n_spin*self.n_orb) )

class diverge_post_patch:
    r'''
    class to read diverge postprocessing files for :math:`N`-patch FRG

    vertices and loops may be None if not explicitly asked for in the
    postprocessing routines. They all are of shape (np, np, np, nb, nb, nb, nb),
    i.e. refer to the patch indices.

    for each interaction channel X (X :math:`\in \{P,C,D\}`), there are the following
    variables:

    :X_chan: boolean that tracks whether this channel is included in the output
    :X_dV: boolean that tracks whether the increment was used (True) or whether
           the vertex at scale was used (False)
    :X_nq: the number of momentum transfer points
    :X_qV: the vertx in 'q representation'. A list of length X_nq with each
           element of the list a tuple (q, k_kp_descr, values, vectors).

            :q: transfer momentum
            :k_kp_descr: array of indices of secondary momenta. Refers to the
                         coarse kmesh. May differ for different q.
            :values: eigenvalues (shape: (nv,) with nv the number of eigenvalues
                     requested) may be None if no eigendecomposition was done
            :vectors: eigenvectors (shape: (nv, len(k_kp_descr), nb, nb)). if
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
            #: full vertex
            self.V = d_data[ 0+d_header[1] : d_header[1]+d_header[2] : 2 ] \
               + 1j* d_data[ 1+d_header[1] : d_header[1]+d_header[2] : 2 ]
        else:
            self.V = None
        if d_header[4] != 0:
            #: particle particle loop
            self.Lp = d_data[ 0+d_header[3] : d_header[3]+d_header[4] : 2 ] \
                + 1j* d_data[ 1+d_header[3] : d_header[3]+d_header[4] : 2 ]
        else:
            self.Lp = None
        if d_header[6] != 0:
            #: particle hole loop
            self.Lm = d_data[ 0+d_header[5] : d_header[5]+d_header[6] : 2 ] \
                + 1j* d_data[ 1+d_header[5] : d_header[5]+d_header[6] : 2 ]
        else:
            self.Lm = None
        if d_header[8] != 0:
            #: increment of the last step
            self.dV = d_data[ 0+d_header[7] : d_header[7]+d_header[8] : 2 ] \
                + 1j* d_data[ 1+d_header[7] : d_header[7]+d_header[8] : 2 ]
        else:
            self.dV = None

        idx = 9
        #: X = P
        self.P_chan = chr(d_header[idx]); idx = idx+1
        #: X = P
        self.P_dV = bool(d_header[idx]); idx = idx+1
        #: X = P
        self.P_nq = d_header[idx]; idx = idx+1
        if d_header[idx+1] > 0:
            #: X = P
            self.P_qV = qmatrices( d_data, d_header[idx], d_header[idx+1], self.P_nq ); idx = idx+2
        else:
            self.P_qV = None

        idx = 14
        #: X = C
        self.C_chan = chr(d_header[idx]); idx = idx+1
        #: X = C
        self.C_dV = bool(d_header[idx]); idx = idx+1
        #: X = C
        self.C_nq = d_header[idx]; idx = idx+1
        if d_header[idx+1] > 0:
            #: X = C
            self.C_qV = qmatrices( d_data, d_header[idx], d_header[idx+1], self.C_nq ); idx = idx+2
        else:
            self.C_qV = None

        idx = 19
        #: X = D
        self.D_chan = chr(d_header[idx]); idx = idx+1
        #: X = D
        self.D_dV = bool(d_header[idx]); idx = idx+1
        #: X = D
        self.D_nq = d_header[idx]; idx = idx+1
        if d_header[idx+1] > 0:
            #: X = D
            self.D_qV = qmatrices( d_data, d_header[idx], d_header[idx+1], self.D_nq ); idx = idx+2
        else:
            self.D_qV = None

        #: diverge tag version (serves as file format version)
        self.version = d_data[ d_header[125] : d_header[125]+d_header[126] ].view(
                dtype=np.byte ).tobytes().decode().rstrip('\x00')

class diverge_post_grid:
    '''
    class to read diverge postprocessing files for grid FRG
    '''
    def __init__(self, fname):
        d_header = numpy_fromfile(fname, dtype=np.longlong, count=64)

        if d_header[0] == DIVERGE_POST_GRID_MAGIC_NUMBER:
            self.valid = True
        else:
            self.valid = False
            return

        d_data = numpy_fromfile(fname)

        #: diverge tag version (serves as file format version)
        self.version = d_data[d_header[60]:d_header[60]+d_header[61]].view(
                dtype=np.byte ).tobytes().decode().rstrip('\x00')

        #: number of kpoints, nk[0]*nk[1]*nk[2] in terms of :c:struct:`diverge_model_t`
        self.nk = d_header[1]
        #: number of bands, n_orb*n_spin in terms of :c:struct:`diverge_model_t`
        self.nb = d_header[2]
        #: number of 'formfactors', some heuristic value
        self.nff = d_header[3]
        #: :math:`SU(2)`?
        self.SU2 = d_header[4]

        #: linearized gap equation number of singular values
        self.lingap_num_ev = d_header[19]
        #: linearized gap equation matrix size
        self.lingap_matrix_size = d_header[20]

        self.file_size = d_header[63]

        #: formfactors (nff,nk)
        formfac = d_data[d_header[5]:d_header[5]+d_header[6]].reshape((self.nff,self.nk,2))
        self.formfac = formfac[:,:,0] + 1j * formfac[:,:,1]

        susc_shape = (self.nk,self.nff,self.nb,self.nb,self.nb,self.nb,2)

        _P_susc = d_data[d_header[7]:d_header[7]+d_header[8]]
        if _P_susc.size != 0:
            #: P susceptibility (nk,nff,nb,nb,nb,nb)
            self.P_susc = _P_susc.reshape(susc_shape)
            self.P_susc = self.P_susc[:,:, :,:,:,:, 0] + 1j * self.P_susc[:,:, :,:,:,:, 1]
        else:
            self.P_susc = None
        _P_mf = d_data[d_header[9]:d_header[9]+d_header[10]]
        if _P_mf.size != 0:
            self.P_mf_U, self.P_mf_V, self.P_mf_S, self.P_mf_EU, self.P_mf_EV = self.unroll_mf_solution( _P_mf )
        else:
            #: P lingap U matrix (lingap_num_ev,lingap_matrix_size)
            self.P_mf_U = None
            #: P lingap V matrix (lingap_num_ev,lingap_matrix_size)
            self.P_mf_V = None
            #: P lingap singular values (lingap_num_ev)
            self.P_mf_S = None
            #: P lingap vertex eigenvectors (lingap_num_ev,lingap_matrix_size)
            self.P_mf_EU = None
            #: P lingap vertex eigenvalues (lingap_num_ev)
            self.P_mf_EV = None

        _C_susc = d_data[d_header[11]:d_header[11]+d_header[12]]
        if _C_susc.size != 0:
            self.C_susc = _C_susc.reshape(susc_shape)
            self.C_susc = self.C_susc[:,:, :,:,:,:, 0] + 1j * self.C_susc[:,:, :,:,:,:, 1]
        else:
            #: C susceptibility (nk,nff,nb,nb,nb,nb)
            self.C_susc = None
        _C_mf = d_data[d_header[13]:d_header[13]+d_header[14]]
        if _C_mf.size != 0:
            self.C_mf_U, self.C_mf_V, self.C_mf_S, self.C_mf_EU, self.C_mf_EV = self.unroll_mf_solution( _C_mf )
        else:
            #: C lingap U matrix (lingap_num_ev,lingap_matrix_size)
            self.C_mf_U = None
            #: C lingap V matrix (lingap_num_ev,lingap_matrix_size)
            self.C_mf_V = None
            #: C lingap singular values (lingap_num_ev)
            self.C_mf_S = None
            #: C lingap vertex eigenvectors (lingap_num_ev,lingap_matrix_size)
            self.C_mf_EU = None
            #: C lingap vertex eigenvalues (lingap_num_ev)
            self.C_mf_EV = None

        _D_susc = d_data[d_header[15]:d_header[15]+d_header[16]]
        if _D_susc.size != 0:
            #: D susceptibility (nk,nff,nb,nb,nb,nb)
            self.D_susc = _D_susc.reshape(susc_shape)
            self.D_susc = self.D_susc[:,:, :,:,:,:, 0] + 1j * self.D_susc[:,:, :,:,:,:, 1]
        else:
            self.D_susc = None
        _D_mf = d_data[d_header[17]:d_header[17]+d_header[18]]
        if _D_mf.size != 0:
            self.D_mf_U, self.D_mf_V, self.D_mf_S, self.D_mf_EU, self.D_mf_EV = self.unroll_mf_solution( _D_mf )
        else:
            #: D lingap U matrix (lingap_num_ev,lingap_matrix_size)
            self.D_mf_U = None
            #: D lingap V matrix (lingap_num_ev,lingap_matrix_size)
            self.D_mf_V = None
            #: D lingap singular values (lingap_num_ev)
            self.D_mf_S = None
            #: D lingap vertex eigenvectors (lingap_num_ev,lingap_matrix_size)
            self.D_mf_EU = None
            #: D lingap vertex eigenvalues (lingap_num_ev)
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

class diverge_post_tu:
    '''
    class to read diverge postprocessing files for tu FRG

    Vertex diagonalisation: For each channel :math:`X \in \{P,C,D\}`, we include

        :Xlen: for each q point the number of stored elements (nkibz)
        :Xoff: for each q point the offset in the array (nkibz)
        :Xtype: for each q point the used diagonalization algorithm (nkibz)
        :Xval: stored Eigenvalues (sum(Xlen))
        :Xvec: stored Eigenvectors (sum(Xlen), n_orb*n_bonds*n_spin**2)

    Lineraized gap solution: For each physical channel Y (sc, mag, charge), we include

        :S_Y: singular values SC lingap (n_sing_val)
        :U_Y: U SC lingap (n_sing_val, n_orbff*n_spin**2)
        :V_Y: V SC lingap (n_sing_val, n_orbff*n_spin**2)

    Susceptibilities: For each channel X, we include the momentum space
    (on-site) susceptibility and the formfactor susceptibilities if enabled in
    the simulation

        :Xsusc: channel momentum on-site susceptibility ([n_orb*n_spin]**4,nkibz)
        :Xsuscff: formfactor resolved susceptibility ([n_orbff*n_spin*n_spin]**2,nkibz)

    any of the above can be 'None' if it was not contained in simulation.

    Full channels: For each channel X, we allow to save the TU vertex at the end
    of the flow as Xchannel. The loops pploop and phloop can be saved as well.
    '''
    def __init__(self, fname):
        d_header = numpy_fromfile(fname, dtype=np.int64, count=128)

        if d_header[0] == DIVERGE_POST_TU_MAGIC_NUMBER and d_header[126] == DIVERGE_POST_TU_MAGIC_NUMBER:
            self.valid = True
        else:
            self.valid = False
            return

        #: number of orbitals (not spin)
        self.n_orb = int(d_header[1])
        #: number of spin degrees of freedom
        self.n_spin = int(d_header[2])
        #: number of momentum points
        self.nk = int(d_header[3])
        #: number of momentum for Hamiltonian
        self.nktot = int(d_header[4])
        #: number of momentum points in the irreducible BZ wedge
        self.nkibz = int(d_header[5])
        #: Exploit :math:`SU2` symmetry?
        self.SU2 = int(d_header[6])
        #: number of orbital+bond combinations
        self.n_orbff = int(d_header[7])
        #: maximal number of bonds of a single site, not necessarily the same for every site
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

        #: map from n_orbff to (o,b) notation, stores o (n_orbff)
        self.mi_to_ofrom = get_array_header( iL, np.int64 )
        #: map from n_orbff to (o,b) notation, stores o+bo (n_orbff)
        self.mi_to_oto = get_array_header( iL, np.longlong )
        #: map from n_orbff to (o,b) notation, stores the beyond unit cell contributiom (n_orbff,3)
        self.mi_to_R = get_array_header( iL, np.longlong ).reshape((-1,3))
        #: number of bonds per site o, needed for correct iteration over tu_ff stored in model (n_orb)
        self.bond_sizes = get_array_header( iL, np.longlong )
        #: offset to the bonds belonging to site o (n_orb)
        self.bond_offsets = get_array_header( iL, np.longlong )
        #: gives the index of the IBZ point in the full PZ (nkibz)
        self.idx_ibz_in_bz = get_array_header( iL, np.longlong )

        #: for each q point the number of stored elements (nkibz)
        self.Plen = get_array_header( iL, np.longlong )
        #: for each q point the offset in the array (nkibz)
        self.Poff = get_array_header( iL, np.longlong )
        #: for each q point the used diagonalization algorithm (nkibz)
        self.Ptype = get_array_header( iL, np.byte )
        #: stored Eigenvalues (sum(Plen))
        self.Pval = get_array_header( iL, np.float64 )
        #: stored Eigenvectors (sum(Plen), n_orbff*n_spin**2)
        self.Pvec = get_array_header( iL, np.complex128 )

        self.Pvec = self.Pvec.reshape((np.sum(self.Plen),n_spin,n_spin,n_orbff))

        #: for each q point the number of stored elements (nkibz)
        self.Clen = get_array_header( iL, np.longlong )
        #: for each q point the offset in the array (nkibz)
        self.Coff = get_array_header( iL, np.longlong )
        #: for each q point the used diagonalization algorithm (nkibz)
        self.Ctype = get_array_header( iL, np.byte )
        #: stored Eigenvalues (sum(Clen))
        self.Cval = get_array_header( iL, np.float64 )
        #: stored Eigenvectors (sum(Clen), n_orbff*n_spin**2)
        self.Cvec = get_array_header( iL, np.complex128 )

        self.Cvec = self.Cvec.reshape((np.sum(self.Clen),n_spin,n_spin,n_orbff))

        #: for each q point the number of stored elements (nkibz)
        self.Dlen = get_array_header( iL, np.longlong )
        #: for each q point the offset in the array (nkibz)
        self.Doff = get_array_header( iL, np.longlong )
        #: for each q point the used diagonalization algorithm (nkibz)
        self.Dtype = get_array_header( iL, np.byte )
        #: stored Eigenvalues (sum(Dlen))
        self.Dval = get_array_header( iL, np.float64 )
        #: stored Eigenvectors (sum(Dlen), n_orbff*n_spin**2)
        self.Dvec = get_array_header( iL, np.complex128 )

        self.Dvec = self.Dvec.reshape((np.sum(self.Dlen),n_spin,n_spin,n_orbff))

        #: singular values sc lingap (n_sing_val)
        self.S_sc = get_array_header( iL, np.float64 )
        self.n_svdP = d_header[iL[0]-1]
        #: U sc lingap (n_sing_val, n_orbff*n_spin**2)
        self.U_sc = get_array_header( iL, np.complex128 )
        self.U_sc = self.U_sc.reshape((self.n_svdP,n_spin,n_spin,n_orbff))
        #: V sc lingap (n_sing_val, n_orbff*n_spin**2)
        self.V_sc = get_array_header( iL, np.complex128 )
        self.V_sc = self.V_sc.reshape((self.n_svdP,n_spin,n_spin,n_orbff))

        #: singular values magnetic lingap (n_sing_val)
        self.S_mag = get_array_header( iL, np.float64 )
        self.n_svdC = d_header[iL[0]-1]
        #: U magnetic lingap (n_sing_val, n_orbff*n_spin**2)
        self.U_mag = get_array_header( iL, np.complex128 )
        self.U_mag = self.U_mag.reshape((self.n_svdC,n_spin,n_spin,n_orbff))
        #: V magnetic lingap (n_sing_val, n_orbff*n_spin**2)
        self.V_mag = get_array_header( iL, np.complex128 )
        self.V_mag = self.V_mag.reshape((self.n_svdC,n_spin,n_spin,n_orbff))

        #: singular values charge lingap (n_sing_val)
        self.S_charge = get_array_header( iL, np.float64 )
        self.n_svdD = d_header[iL[0]-1]
        #: U charge lingap (n_sing_val, n_orbff*n_spin**2)
        self.U_charge = get_array_header( iL, np.complex128 )
        self.U_charge = self.U_charge.reshape((self.n_svdD,n_spin,n_spin,n_orbff))
        #: V charge lingap (n_sing_val, n_orbff*n_spin**2)
        self.V_charge = get_array_header( iL, np.complex128 )
        self.V_charge = self.V_charge.reshape((self.n_svdD,n_spin,n_spin,n_orbff))

        susc_shape = (n_spin,n_orb,n_spin,n_orb,n_spin,n_orb,n_spin,n_orb,self.nk)

        #: Pair-Pair susceptibility ([n_orb*n_spin]**4,nkibz)
        self.Psusc = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Psusc = self.Psusc.reshape(susc_shape)
        #: magnetic/crossed PH susceptibility ([n_orb*n_spin]**4,nkibz)
        self.Csusc = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Csusc = self.Csusc.reshape(susc_shape)
        #: charge/direct PH susceptibility ([n_orb*n_spin]**4,nkibz)
        self.Dsusc = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Dsusc = self.Dsusc.reshape(susc_shape)

        susc_shape = (n_spin*n_spin,n_orbff,n_spin*n_spin,n_orbff,self.nk)
        #: formfactor resolved Pair-Pair susceptibility ([n_orbff*n_spin*n_spin]**2,nkibz)
        self.Psusc_ff = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Psusc_ff = self.Psusc_ff.reshape(susc_shape)
        #: formfactor resolved magnetic/crossed PH susceptibility ([n_orbff*n_spin*n_spin]**2,nkibz)
        self.Csusc_ff = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Csusc_ff = self.Csusc_ff.reshape(susc_shape)
        #: formfactor resolved charge/direct PH susceptibility (n_orbff*n_spin*n_spin]**2,nkibz)
        self.Dsusc_ff = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Dsusc_ff = self.Dsusc_ff.reshape(susc_shape)

        #: Self-energy at the critical scale ([n_orb*n_spin]**2,nktot)
        self.selfenergy = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.selfenergy = self.selfenergy.reshape(self.nktot,n_spin,n_orb,n_spin,n_orb)

        #: P channel on full PZ
        self.Pchannel = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Pchannel = self.Pchannel.reshape(susc_shape)
        #: C channel on full PZ
        self.Cchannel = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Cchannel = self.Cchannel.reshape(susc_shape)
        #: D channel on full PZ
        self.Dchannel = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.Dchannel = self.Dchannel.reshape(susc_shape)
        #: non interacting PP-loop channel on full PZ
        self.pploop = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.pploop = self.pploop.reshape(susc_shape)
        #: non interacting PH-loop channel on full PZ
        self.phloop = get_array_header( iL, np.complex128 )
        if(d_header[iL[0]-1] > 0):
            self.phloop = self.phloop.reshape(susc_shape)

        symm_shape = (self.n_sym,n_spin,n_spin,n_orbff)
        #: length of symmetry maps
        self.symm_o2m_len = get_array_header( iL, np.longlong )
        if(d_header[iL[0]-1] > 0):
            self.symm_o2m_len = self.symm_o2m_len.reshape(symm_shape)
        #: offsets of symmetry maps
        self.symm_o2m_off = get_array_header( iL, np.longlong )
        if(d_header[iL[0]-1] > 0):
            self.symm_o2m_off = self.symm_o2m_off.reshape(symm_shape)
        #: index maps
        self.symm_o2m_idx_map = get_array_header( iL, np.longlong )
        #: complex prefactor for P
        self.symm_o2m_Ppref = get_array_header( iL, np.complex128 )
        #: complex prefactor for C
        self.symm_o2m_Cpref = get_array_header( iL, np.complex128 )

        #: diverge tag version (serves as file format version)
        self.version = get_array_header( iL, np.byte ).tobytes().decode()

        #: symmetry mapping of full BZ to IBZ (nk)
        self.kmaps_to = get_array_header( iL, np.longlong )
        #: multiindex to tu_ff index mapping; useful for analysis with model.tu_ff (n_orbff)
        self.mi_to_tuffidx = get_array_header( iL, np.longlong )

def read( fname ):
    r"""
    function to read model files as well as post processing files. returns the
    corresponding class (:class:`diverge_model`, :class:`diverge_post_patch`,
    :class:`diverge_post_grid`, or :class:`diverge_post_tu`). Makes use of to
    the file format specifications given in
    :c:func:`diverge_postprocess_and_write` and :c:func:`diverge_model_to_file`;
    discerning the different file types by their "magic numbers".

    :param fname: file name to read from (typically ```XXX.dvg```)
    """
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

def bandstructure_bands( model ):
    r"""
    return the bands array (nk,nb) from the band  structure obtained from a
    :class:`diverge_model`. Useful for plotting (for example see
    :func:`bandstructure_xvals`).
    """
    return model.bandstructure[:,:-3]

def bandstructure_kpts( model ):
    r"""
    return the momentum points (nk,3) from the band structure obtained from a
    :class:`diverge_model`. Useful for plotting and post-processing.
    """
    return model.bandstructure[:,-3:]

def bandstructure_xvals( model ):
    r"""
    calculate the differential distance between the momentum points from the band
    structure obtained from a :class:`diverge_model`. Useful for plotting in
    conjuction with :func:`bandstructure_bands` and :func:`bandstructure_ticks`.

    Example:

    .. sourcecode:: python

        import diverge.output as do
        import matplotlib.pyplot as plt
        M = do.read("model.dvg")
        b = do.bandstructure_bands(M)
        x = do.bandstructure_xvals(M)
        plt.plot( x, b, color='black' )
        plt.xticks( do.bandstructure_ticks(M) )
        # need to set the labels manually as they are not known to the model
        plt.show()
    """
    K = bandstructure_kpts(model)
    vals = np.concatenate( [[0], np.sqrt((np.diff(K,axis=0)**2).sum(axis=1))] )
    if model.npath == -1 and not model.kf_ibz_path is None:
        n_per_segment = model.kf_ibz_path[0]
        offsets = np.cumsum( n_per_segment )
        offset_zero = offsets[n_per_segment == 0]
        vals[offset_zero] = 0.0
    return np.cumsum(vals)

def bandstructure_ticks( model ):
    r"""
    returns the xticks used for band structure plots from a
    :class:`diverge_model`. Useful in conjuction with
    :func:`bandstructure_xvals`.
    """
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

def ibz_path_vals( kf_ibz_path ):
    r"""
    returns the correctly spaced cumulative distance values associated to an IBZ
    path object (as returned in :attr:`diverge_model.kc_ibz_path` and
    :attr:`diverge_model.kf_ibz_path`).
    """
    K = kf_ibz_path[2]
    vals = np.concatenate( [[0], np.sqrt((np.diff(K,axis=0)**2).sum(axis=1))] )
    n_per_segment = kf_ibz_path[0]
    offsets = np.cumsum( n_per_segment )
    offset_zero = offsets[n_per_segment == 0]
    vals[offset_zero] = 0.0
    return np.cumsum(vals)

def ibz_path_ticks( kf_ibz_path ):
    r"""
    returns the correctly spaced ticks corresponding to the cumulative distance
    returned by :func:`ibz_path_vals`.
    """
    n_per_segment = kf_ibz_path[0]
    x = ibz_path_vals(kf_ibz_path)
    return list(x[np.concatenate( [[0], np.cumsum( n_per_segment ) - 1] )])

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
