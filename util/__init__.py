# Namespace: dvg
# import diverge::
# Wrapper for diverge.h, generated with some help of ctypesgen. It is heavily
# based on wrapping C arrays as numpy structured arrays. Therefore, some
# documentation for these (and how to use them from <dvg>) is included <below at
# Structured Arrays>.
#
# The documentation only includes those functions that exist *on top* of the <C
# library at libdivERGe.so>. Everything else is documented <there at
# libdivERGe.so>. For usage examples, see the python examples shipped with
# <diverge.tar.gz at
# https://git.rwth-aachen.de/frg/diverge/-/raw/master/public/releases/v0.2/divERGe.tar.gz>
# (or found in the <examples/ directory of the repo at
# https://git.rwth-aachen.de/frg/diverge/-/tree/master/examples>).
#
# Note::
# on linux, the python library will search for libdivERGe.so in your
# LD_LIBRARY_PATH _with precedence_ over the shipped version (located somewhere
# in your PYTHONPATH). This makes it easy to swap out the backend by pointing
# LD_LIBRARY_PATH to a location with the desired libdivERGe.so:
# === Code ===
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/gpu/version/ python script.py
# ============

from diverge.helpers import *
from diverge.helpers import String, ReturnString, _libs
from ctypes import *

# Function: info
# print information on the shared library
def info():
    try:
        info_str = str(_libs["divERGe"].access['cdecl'])
    except:
        info_str = "libdivERGe.so not found, or unable to use it."
    try:
        mpi_py_eprint( info_str )
    except:
        pass
    return info_str

if not _libs["divERGe"] is None:

    class struct_complex128_t(Structure):
        pass
    struct_complex128_t.__slots__ = [
        'x',
        'y',
    ]
    struct_complex128_t._fields_ = [
        ('x', c_double),
        ('y', c_double),
    ]
    complex128_t = struct_complex128_t
    gf_complex_t = complex128_t
    index_t = c_int64

    # MPI functions
    init_ = _libs["divERGe"].get("diverge_init", "cdecl")
    init_.argtypes = [POINTER(c_int), POINTER(POINTER(POINTER(c_char)))]
    init_.restype = None
    def init( p_argc=None, p_argv=None ):
        init_( p_argc, p_argv )

    finalize = _libs["divERGe"].get("diverge_finalize", "cdecl")
    finalize.argtypes = []
    finalize.restype = None

    embed = _libs["divERGe"].get("diverge_embed", "cdecl")
    embed.argtypes = [c_voidp]
    embed.restype = None

    reset = _libs["divERGe"].get("diverge_reset", "cdecl")
    reset.argtypes = []
    reset.restype = None

    mpi_exit = _libs["divERGe"].get("diverge_mpi_exit", "cdecl")
    mpi_exit.argtypes = [c_int]
    mpi_exit.restype = None

    mpi_wtime = _libs["divERGe"].get("diverge_mpi_wtime", "cdecl")
    mpi_wtime.argtypes = []
    mpi_wtime.restype = c_double

    mpi_get_comm = _libs["divERGe"].get("diverge_mpi_get_comm", "cdecl")
    mpi_get_comm.argtypes = []
    mpi_get_comm.restype = c_voidp

    mpi_distribute = _libs["divERGe"].get("diverge_mpi_distribute", "cdecl")
    mpi_distribute.argtypes = [index_t]
    mpi_distribute.restype = POINTER(index_t)

    mpi_barrier = _libs["divERGe"].get("diverge_mpi_barrier", "cdecl")
    mpi_barrier.argtypes = []
    mpi_barrier.restype = None

    mpi_comm_size = _libs["divERGe"].get("diverge_mpi_comm_size", "cdecl")
    mpi_comm_size.argtypes = []
    mpi_comm_size.restype = c_int

    mpi_comm_rank = _libs["divERGe"].get("diverge_mpi_comm_rank", "cdecl")
    mpi_comm_rank.argtypes = []
    mpi_comm_rank.restype = c_int

    mpi_allreduce_double_max = _libs["divERGe"].get("diverge_mpi_allreduce_double_max", "cdecl")
    mpi_allreduce_double_max.argtypes = [POINTER(None), POINTER(None), c_int]
    mpi_allreduce_double_max.restype = None

    mpi_allreduce_double_sum = _libs["divERGe"].get("diverge_mpi_allreduce_double_sum", "cdecl")
    mpi_allreduce_double_sum.argtypes = [POINTER(None), POINTER(None), c_int]
    mpi_allreduce_double_sum.restype = None

    mpi_allreduce_complex_sum = _libs["divERGe"].get("diverge_mpi_allreduce_complex_sum", "cdecl")
    mpi_allreduce_complex_sum.argtypes = [POINTER(None), POINTER(None), c_int]
    mpi_allreduce_complex_sum.restype = None

    mpi_allgather_index = _libs["divERGe"].get("diverge_mpi_allgather_index", "cdecl")
    mpi_allgather_index.argtypes = [POINTER(None), POINTER(None), c_int]
    mpi_allgather_index.restype = None

    mpi_allgather_double = _libs["divERGe"].get("diverge_mpi_allgather_double", "cdecl")
    mpi_allgather_double.argtypes = [POINTER(None), POINTER(None), c_int]
    mpi_allgather_double.restype = None

    mpi_send_double = _libs["divERGe"].get("diverge_mpi_send_double", "cdecl")
    mpi_send_double.argtypes = [POINTER(None), c_int, c_int, c_int]
    mpi_send_double.restype = None

    mpi_recv_double = _libs["divERGe"].get("diverge_mpi_recv_double", "cdecl")
    mpi_recv_double.argtypes = [POINTER(None), c_int, c_int, c_int]
    mpi_recv_double.restype = None

    mpi_gatherv_cdoub = _libs["divERGe"].get("diverge_mpi_gatherv_cdoub", "cdecl")
    mpi_gatherv_cdoub.argtypes = [POINTER(None), c_int, POINTER(None), POINTER(c_int), POINTER(c_int), c_int]
    mpi_gatherv_cdoub.restype = None

    mpi_write_cdoub_to_file = _libs["divERGe"].get("diverge_mpi_write_cdoub_to_file", "cdecl")
    mpi_write_cdoub_to_file.argtypes = [String, POINTER(None), c_int, c_int]
    mpi_write_cdoub_to_file.restype = None

    mpi_alltoallv_bytes = _libs["divERGe"].get("diverge_mpi_alltoallv_bytes", "cdecl")
    mpi_alltoallv_bytes.argtypes = [POINTER(None), POINTER(index_t), POINTER(index_t), POINTER(None), POINTER(index_t), POINTER(index_t), index_t]
    mpi_alltoallv_bytes.restype = None

    mpi_alltoallv_complex = _libs["divERGe"].get("diverge_mpi_alltoallv_complex", "cdecl")
    mpi_alltoallv_complex.argtypes = [POINTER(complex128_t), POINTER(c_int), POINTER(c_int), POINTER(complex128_t), POINTER(c_int), POINTER(c_int)]
    mpi_alltoallv_complex.restype = None

    mpi_allgatherv = _libs["divERGe"].get("diverge_mpi_allgatherv", "cdecl")
    mpi_allgatherv.argtypes = [POINTER(complex128_t), POINTER(c_int), POINTER(c_int)]
    mpi_allgatherv.restype = None

    mpi_max = _libs["divERGe"].get("diverge_mpi_max", "cdecl")
    mpi_max.argtypes = [POINTER(c_double)]
    mpi_max.restype = c_double

    mpi_bcast_bytes = _libs["divERGe"].get("diverge_mpi_bcast_bytes", "cdecl")
    mpi_bcast_bytes.argtypes = [POINTER(None), c_int, c_int]
    mpi_bcast_bytes.restype = None

    mpi_gather_double = _libs["divERGe"].get("diverge_mpi_gather_double", "cdecl")
    mpi_gather_double.argtypes = [POINTER(c_double), c_int, POINTER(c_double), c_int, c_int]
    mpi_gather_double.restype = None

    # Threading
    omp_num_threads = _libs["divERGe"].get("diverge_omp_num_threads", "cdecl")
    omp_num_threads.argtypes = None
    omp_num_threads.restype = c_int

    force_thread_limit = _libs["divERGe"].get("diverge_force_thread_limit", "cdecl")
    force_thread_limit.argtypes = [c_int]
    force_thread_limit.restype = None

    # MPI logging
    mpi_loglevel_set = _libs["divERGe"].get("mpi_loglevel_set", "cdecl")
    mpi_loglevel_set.argtypes = [c_int]
    mpi_loglevel_set.restype = None

    mpi_log_set_colors = _libs["divERGe"].get("mpi_log_set_colors", "cdecl")
    mpi_log_set_colors.argtypes = [c_int]
    mpi_log_set_colors.restype = None

    mpi_log_get_colors = _libs["divERGe"].get("mpi_log_get_colors", "cdecl")
    mpi_log_get_colors.argtypes = None
    mpi_log_get_colors.restype = c_int

    mpi_log_control = _libs["divERGe"].get("mpi_log_control", "cdecl")
    mpi_log_control.argtypes = [c_int]
    mpi_log_control.restype = None

    mpi_py_print = _libs["divERGe"].get("mpi_py_print", "cdecl")
    mpi_py_print.argtypes = [String]
    mpi_py_print.restype = None

    mpi_py_eprint = _libs["divERGe"].get("mpi_py_eprint", "cdecl")
    mpi_py_eprint.argtypes = [String]
    mpi_py_eprint.restype = None

    mpi_py_print_all = _libs["divERGe"].get("mpi_py_print_all", "cdecl")
    mpi_py_print_all.argtypes = [String]
    mpi_py_print_all.restype = None

    mpi_py_eprint_all = _libs["divERGe"].get("mpi_py_eprint_all", "cdecl")
    mpi_py_eprint_all.argtypes = [String]
    mpi_py_eprint_all.restype = None

    # Compilation Status
    compilation_status = _libs["divERGe"].get("diverge_compilation_status", "cdecl")
    compilation_status.argtypes = []
    compilation_status.restype = None

    compilation_status = _libs["divERGe"].get("diverge_compilation_status", "cdecl")
    compilation_status.argtypes = []
    compilation_status.restype = None

    compilation_status_mpi = _libs["divERGe"].get("diverge_compilation_status_mpi", "cdecl")
    compilation_status_mpi.argtypes = []
    compilation_status_mpi.restype = c_int
    compilation_status_cuda = _libs["divERGe"].get("diverge_compilation_status_cuda", "cdecl")
    compilation_status_cuda.argtypes = [String]
    compilation_status_cuda.restype = c_int
    compilation_status_version = _libs["divERGe"].get("diverge_compilation_status_version", "cdecl")
    compilation_status_version.argtypes = [String]
    compilation_status_version.restype = c_int
    compilation_status_numbers = _libs["divERGe"].get("diverge_compilation_status_numbers", "cdecl")
    compilation_status_numbers.argtypes = []
    compilation_status_numbers.restype = c_int

    # License
    license = _libs["divERGe"].get("diverge_license", "cdecl")
    license.argtypes = []
    license.restype = String

    license_print = _libs["divERGe"].get("diverge_license_print", "cdecl")
    license_print.argtypes = []
    license_print.restype = None

    # Symmetry Generator
    real_harmonics_t = c_int
    orb_s = 0
    orb_pm1 = 1
    orb_p_0 = 2
    orb_p_1 = 3
    orb_py = 1
    orb_pz = 2
    orb_px = 3
    orb_dm2 = 4
    orb_dm1 = 5
    orb_d0 = 6
    orb_d1 = 7
    orb_d2 = 8
    orb_dxy = 4
    orb_dyz = 5
    orb_dz2 = 6
    orb_dxz = 7
    orb_dx2y2 = 8
    orb_fm3 = 9
    orb_fm2 = 10
    orb_fm1 = 11
    orb_f_0 = 12
    orb_f_1 = 13
    orb_f_2 = 14
    orb_f_3 = 15
    orb_gm4 = 16
    orb_gm3 = 17
    orb_gm2 = 18
    orb_gm1 = 19
    orb_g_0 = 20
    orb_g_1 = 21
    orb_g_2 = 22
    orb_g_3 = 23
    orb_g_4 = 24
    MAX_ORBS_PER_SITE = 20
    class struct_sym_op_t(Structure):
        pass
    sym_op_t = struct_sym_op_t
    struct_sym_op_t.__slots__ = [
        'type',
        'normal_vector',
        'angle',
    ]
    struct_sym_op_t._fields_ = [
        ('type', c_char),
        ('normal_vector', c_double * int(3)),
        ('angle', c_double),
    ]

    class struct_site_descr_t(Structure):
        pass
    site_descr_t = struct_site_descr_t
    struct_site_descr_t.__slots__ = [
        'amplitude',
        'function',
        'n_functions',
        'xaxis',
        'yaxis',
    ]
    struct_site_descr_t._fields_ = [
        ('amplitude', complex128_t * int(MAX_ORBS_PER_SITE)),
        ('function', real_harmonics_t * int(MAX_ORBS_PER_SITE)),
        ('n_functions', index_t),
        ('xaxis', c_double * int(3*MAX_ORBS_PER_SITE)),
        ('yaxis', c_double * int(3*MAX_ORBS_PER_SITE)),
    ]

    generate_symm_trafo_ = _libs["divERGe"].get("diverge_generate_symm_trafo", "cdecl")
    generate_symm_trafo_.argtypes = [ index_t, c_voidp, index_t, c_voidp, index_t, c_voidp, c_voidp ]
    generate_symm_trafo_.restype = None

    # Function: generate_symm_trafo
    # python version of the symmetry transformation
    #
    # Parameters:
    # n_spin - number of spins
    # orbs - np.array( ..., dtype=site_descr_t )
    # syms - np.array( ..., dtype=sym_op_t )
    #
    # Returns:
    # rs_trafo - (3,3) double array
    # orb_trafo - (n_spin*orbs.size, n_spin*orbs.size) complex128_t array
    def generate_symm_trafo( n_spin, orbs, syms ):
        rs_trafo = np.zeros( (3,3), dtype=np.float64 )
        orb_trafo = np.zeros( (n_spin*orbs.size, n_spin*orbs.size), dtype=np.complex128 )

        generate_symm_trafo_( n_spin, orbs.ctypes.data, orbs.size,
                             syms.ctypes.data, syms.size, rs_trafo.ctypes.data,
                             orb_trafo.ctypes.data )
        return rs_trafo, orb_trafo

    # Model

    # structures typedefs
    class struct_rs_hopping_t(Structure):
        pass
    rs_hopping_t = struct_rs_hopping_t

    class struct_rs_vertex_t(Structure):
        pass
    rs_vertex_t = struct_rs_vertex_t

    class struct_mom_patching_t(Structure):
        pass
    mom_patching_t = struct_mom_patching_t

    class struct_model_t(Structure):
        pass
    model_t = struct_model_t

    class struct_internals_t(Structure):
        pass
    internals_t = struct_internals_t

    class struct_tu_formfactor_t(Structure):
        pass
    tu_formfactor_t = struct_tu_formfactor_t

    # function pointers for callbacks
    hamiltonian_generator_t = CFUNCTYPE(UNCHECKED(None), POINTER(model_t), POINTER(complex128_t))
    # greensfunction return values
    enum_greensfunc_op_t = c_int
    greensfunc_op_cpu = 0
    greensfunc_op_gpu = (greensfunc_op_cpu + 1)
    greensfunc_op_t = enum_greensfunc_op_t

    greensfunc_generator_t = CFUNCTYPE(UNCHECKED(greensfunc_op_t), POINTER(model_t), complex128_t, POINTER(gf_complex_t))

    channel_vertex_generator_t = CFUNCTYPE(UNCHECKED(c_int), POINTER(model_t), c_char, POINTER(complex128_t))

    full_vertex_generator_t = CFUNCTYPE(UNCHECKED(None), POINTER(model_t), index_t, index_t, index_t, POINTER(complex128_t))

    # structures
    struct_rs_hopping_t.__slots__ = [
        'R',
        'o1',
        'o2',
        's1',
        's2',
        't',
    ]
    struct_rs_hopping_t._fields_ = [
        ('R', index_t * int(3)),
        ('o1', index_t),
        ('o2', index_t),
        ('s1', index_t),
        ('s2', index_t),
        ('t', complex128_t),
    ]

    struct_rs_vertex_t.__slots__ = [
        'chan',
        'R',
        'o1',
        'o2',
        's1',
        's2',
        's3',
        's4',
        'V',
    ]
    struct_rs_vertex_t._fields_ = [
        ('chan', c_char),
        ('R', index_t * int(3)),
        ('o1', index_t),
        ('o2', index_t),
        ('s1', index_t),
        ('s2', index_t),
        ('s3', index_t),
        ('s4', index_t),
        ('V', complex128_t),
    ]

    struct_mom_patching_t.__slots__ = [
        'n_patches',
        'patches',
        'weights',
        'p_count',
        'p_displ',
        'p_map',
        'p_weights',
    ]
    struct_mom_patching_t._fields_ = [
        ('n_patches', index_t),
        ('patches', POINTER(index_t)),
        ('weights', POINTER(c_double)),
        ('p_count', POINTER(index_t)),
        ('p_displ', POINTER(index_t)),
        ('p_map', POINTER(index_t)),
        ('p_weights', POINTER(c_double)),
    ]

    struct_tu_formfactor_t.__slots__ = [
        'R',
        'ofrom',
        'oto',
        'd',
        'ffidx',
    ]
    struct_tu_formfactor_t._fields_ = [
        ('R', index_t * int(3)),
        ('ofrom', index_t),
        ('oto', index_t),
        ('d', c_double),
        ('ffidx', index_t),
    ]

    struct_model_t.__slots__ = [
        'name',
        'nk',
        'nkf',
        'patching',
        'n_ibz_path',
        'ibz_path',
        'n_orb',
        'lattice',
        'positions',
        'n_sym',
        'orb_symmetries',
        'rs_symmetries',
        'n_hop',
        'hop',
        'hfill',
        'SU2',
        'n_spin',
        'n_vert',
        'vert',
        'tu_ff',
        'n_tu_ff',
        'n_vert_chan',
        'vfill',
        'ffill',
        'gfill',
        'gproj',
        'data',
        'nbytes_data',
        'data_destructor',
        'internals',
    ]
    struct_model_t._fields_ = [
        ('name', c_char * int(1024)),
        ('nk', index_t * int(3)),
        ('nkf', index_t * int(3)),
        ('patching', POINTER(struct_mom_patching_t)),
        ('n_ibz_path', index_t),
        ('ibz_path', (c_double * int(3)) * int(32768)),
        ('n_orb', index_t),
        ('lattice', (c_double * int(3)) * int(3)),
        ('positions', (c_double * int(3)) * int(32768)),
        ('n_sym', index_t),
        ('orb_symmetries', c_voidp),
        ('rs_symmetries', ((c_double * int(3)) * int(3)) * int(256)),
        ('n_hop', index_t),
        ('hop', c_voidp),
        ('hfill', hamiltonian_generator_t),
        ('SU2', c_int),
        ('n_spin', index_t),
        ('n_vert', index_t),
        ('vert', c_voidp),
        ('tu_ff', c_voidp),
        ('n_tu_ff', index_t),
        ('n_vert_chan', index_t * int(3)),
        ('vfill', channel_vertex_generator_t),
        ('ffill', full_vertex_generator_t),
        ('gfill', greensfunc_generator_t),
        ('grpoj', greensfunc_generator_t),
        ('data', String),
        ('nbytes_data', index_t),
        ('data_destructor', CFUNCTYPE(UNCHECKED(None), POINTER(c_char))),
        ('internals', POINTER(internals_t)),
    ]

    # function definitions
    diverge_hamilton_generator_default = _libs["divERGe"].get("diverge_hamilton_generator_default", "cdecl")
    diverge_hamilton_generator_default.argtypes = [POINTER(model_t), POINTER(complex128_t)]
    diverge_hamilton_generator_default.restype = None

    diverge_hamilton_generator_add = _libs["divERGe"].get("diverge_hamilton_generator_add", "cdecl")
    diverge_hamilton_generator_add.argtypes = [POINTER(model_t), POINTER(complex128_t)]
    diverge_hamilton_generator_add.restype = None

    diverge_greensfunc_generator_default = _libs["divERGe"].get("diverge_greensfunc_generator_default", "cdecl")
    diverge_greensfunc_generator_default.argtypes = [POINTER(model_t), complex128_t, POINTER(gf_complex_t)]
    diverge_greensfunc_generator_default.restype = greensfunc_op_t

    diverge_channel_vertex_generator_default = _libs["divERGe"].get("diverge_channel_vertex_generator_default", "cdecl")
    diverge_channel_vertex_generator_default.argtypes = [POINTER(model_t), c_char, POINTER(complex128_t)]
    diverge_channel_vertex_generator_default.restype = c_int

    model_init = _libs["divERGe"].get("diverge_model_init", "cdecl")
    model_init.argtypes = []
    model_init.restype = POINTER(model_t)

    read_fplo = _libs["divERGe"].get("diverge_read_fplo", "cdecl")
    read_fplo.argtypes = [String]
    read_fplo.restype = POINTER(model_t)

    read_W90 = _libs["divERGe"].get("diverge_read_W90_C", "cdecl")
    read_W90.argtypes = [c_char_p,index_t,POINTER(index_t),POINTER(index_t)]
    read_W90.restype = c_void_p
    # Function: read_W90_PY
    # returns a numpy array of <rs_hopping_t> with the hoppings from W90 file
    # and its length
    #
    # Parameters:
    # fname - filename to read W90 data from (usually ..._hr.dat)
    # nspin - default value 0 amounts to SU(2) symmetric model. if nspin != 0,
    #         abs(nspin) = (S+1/2)*2 with S the physical spin (i.e., for S=1/2 we
    #         have abs(nspin)=2). the sign determines whether the spin index is the
    #         one which increases memory slowly (negative) for fast (positive)
    def read_W90_PY( fname, nspin ):
        length = index_t()
        n_orb = index_t(0)
        ptr = POINTER(rs_hopping_t)
        ptr = read_W90( fname.encode('utf-8'), nspin, byref(length), byref(n_orb) )
        pts_ary = view_array( ptr, dtype=rs_hopping_t, shape=(np.array(length),) )
        pts_cpy = np.copy( pts_ary )
        mem_free( ptr )
        return pts_cpy

    rs_hopping_to_supercell = _libs["divERGe"].get("rs_hopping_to_supercell", "cdecl")
    rs_hopping_to_supercell.argtypes = [c_voidp, c_voidp]
    rs_hopping_to_supercell.restype = None

    rs_hopping_to_fractcell = _libs["divERGe"].get("rs_hopping_to_fractcell", "cdecl")
    rs_hopping_to_fractcell.argtypes = [c_voidp, c_voidp, c_voidp, c_voidp]
    rs_hopping_to_fractcell.restype = None

    fukui = _libs["divERGe"].get("diverge_fukui", "cdecl")
    fukui.argtypes = [POINTER(model_t), c_voidp, index_t, c_voidp]
    fukui.restype = c_voidp

    kmesh_to_bands = _libs["divERGe"].get("diverge_kmesh_to_bands", "cdecl")
    kmesh_to_bands.argtypes = [POINTER(model_t), POINTER(POINTER(index_t)), POINTER(index_t)]
    kmesh_to_bands.restype = POINTER(index_t)

    kmesh_to_bands_crs = _libs["divERGe"].get("diverge_kmesh_to_bands_crs", "cdecl")
    kmesh_to_bands_crs.argtypes = [POINTER(model_t), POINTER(POINTER(index_t)), POINTER(index_t)]
    kmesh_to_bands_crs.restype = POINTER(index_t)

    model_free = _libs["divERGe"].get("diverge_model_free", "cdecl")
    model_free.argtypes = [POINTER(model_t)]
    model_free.restype = None

    # Memory allocation routines
    mem_alloc_rs_hopping_t = _libs["divERGe"].get("diverge_mem_alloc_rs_hopping_t", "cdecl")
    mem_alloc_rs_hopping_t.argtypes = [index_t]
    mem_alloc_rs_hopping_t.restype = POINTER(rs_hopping_t)

    mem_alloc_rs_vertex_t = _libs["divERGe"].get("diverge_mem_alloc_rs_vertex_t", "cdecl")
    mem_alloc_rs_vertex_t.argtypes = [index_t]
    mem_alloc_rs_vertex_t.restype = POINTER(rs_vertex_t)

    mem_alloc_tu_formfactor_t = _libs["divERGe"].get("diverge_mem_alloc_tu_formfactor_t", "cdecl")
    mem_alloc_tu_formfactor_t.argtypes = [index_t]
    mem_alloc_tu_formfactor_t.restype = POINTER(tu_formfactor_t)

    mem_alloc_complex128_t = _libs["divERGe"].get("diverge_mem_alloc_complex128_t", "cdecl")
    mem_alloc_complex128_t.argtypes = [index_t]
    mem_alloc_complex128_t.restype = POINTER(c_double)

    mem_free = _libs["divERGe"].get("diverge_mem_free", "cdecl")
    mem_free.argtypes = [POINTER(None)]
    mem_free.restype = None

    # Model validation and internals
    model_validate = _libs["divERGe"].get("diverge_model_validate", "cdecl")
    model_validate.argtypes = [POINTER(model_t)]
    model_validate.restype = c_int

    model_internals_common = _libs["divERGe"].get("diverge_model_internals_common", "cdecl")
    model_internals_common.argtypes = [POINTER(model_t)]
    model_internals_common.restype = None

    model_internals_grid = _libs["divERGe"].get("diverge_model_internals_grid", "cdecl")
    model_internals_grid.argtypes = [POINTER(model_t)]
    model_internals_grid.restype = None

    model_internals_patch = _libs["divERGe"].get("diverge_model_internals_patch", "cdecl")
    model_internals_patch.argtypes = [POINTER(model_t), index_t]
    model_internals_patch.restype = None

    model_internals_reset = _libs["divERGe"].get("diverge_model_internals_reset", "cdecl")
    model_internals_reset.argtypes = [POINTER(model_t)]
    model_internals_reset.restype = None

    max_dist_iobi = _libs["divERGe"].get("diverge_max_dist_iobi", "cdecl")
    max_dist_iobi.argtypes = None
    max_dist_iobi.restype = c_double

    model_internals_tu = _libs["divERGe"].get("diverge_model_internals_tu", "cdecl")
    model_internals_tu.argtypes = [POINTER(model_t), c_double]
    model_internals_tu.restype = None

    class ArgUnion(Union):
        _fields_ = [("np_ibz", index_t),
                    ("max_dist", c_double)]
    model_internals_any_ = _libs["divERGe"].get("diverge_model_internals_any", "cdecl")
    model_internals_any_.argtypes = [POINTER(model_t), String, ArgUnion]
    model_internals_any_.restype = None
    # Function: model_internals_any_PY
    # wraps <diverge_model_internals_any> with varargs substituted by kwargs
    # np_ibz or max_dist.
    def model_internals_any_PY( model, mode, np_ibz=None, max_dist=None ):
        arg = ArgUnion()
        if not np_ibz is None:
            arg.np_ibz = np_ibz
        if not max_dist is None:
            arg.max_dist = max_dist
        model_internals_any_( model, mode, arg )

    model_internals_get_E = _libs["divERGe"].get("diverge_model_internals_get_E", "cdecl")
    model_internals_get_E.argtypes = [POINTER(model_t)]
    model_internals_get_E.restype = POINTER(c_double)

    model_internals_get_U = _libs["divERGe"].get("diverge_model_internals_get_U", "cdecl")
    model_internals_get_U.argtypes = [POINTER(model_t)]
    model_internals_get_U.restype = POINTER(complex128_t)

    model_internals_get_H = _libs["divERGe"].get("diverge_model_internals_get_H", "cdecl")
    model_internals_get_H.argtypes = [POINTER(model_t)]
    model_internals_get_H.restype = POINTER(complex128_t)

    model_internals_get_kmesh = _libs["divERGe"].get("diverge_model_internals_get_kmesh", "cdecl")
    model_internals_get_kmesh.argtypes = [POINTER(model_t)]
    model_internals_get_kmesh.restype = POINTER(c_double)

    model_internals_get_kfmesh = _libs["divERGe"].get("diverge_model_internals_get_kfmesh", "cdecl")
    model_internals_get_kfmesh.argtypes = [POINTER(model_t)]
    model_internals_get_kfmesh.restype = POINTER(c_double)

    model_internals_get_greens = _libs["divERGe"].get("diverge_model_internals_get_greens", "cdecl")
    model_internals_get_greens.argtypes = [POINTER(model_t)]
    model_internals_get_greens.restype = POINTER(gf_complex_t)

    # Filling
    model_get_filling = _libs["divERGe"].get("diverge_model_get_filling", "cdecl")
    model_get_filling.argtypes = [POINTER(model_t), POINTER(c_double), index_t]
    model_get_filling.restype = c_double

    model_set_filling = _libs["divERGe"].get("diverge_model_set_filling", "cdecl")
    model_set_filling.argtypes = [POINTER(model_t), POINTER(c_double), index_t, c_double]
    model_set_filling.restype = c_double

    model_set_chempot = _libs["divERGe"].get("diverge_model_set_chempot", "cdecl")
    model_set_chempot.argtypes = [POINTER(model_t), POINTER(c_double), index_t, c_double]
    model_set_chempot.restype = None

    # Flow step
    class struct_flow_step_t(Structure):
        pass
    flow_step_t = struct_flow_step_t

    flow_step_init = _libs["divERGe"].get("diverge_flow_step_init", "cdecl")
    flow_step_init.argtypes = [POINTER(model_t), String, String]
    flow_step_init.restype = POINTER(flow_step_t)

    flow_step_init_any = _libs["divERGe"].get("diverge_flow_step_init_any", "cdecl")
    flow_step_init_any.argtypes = [POINTER(model_t), String]
    flow_step_init_any.restype = POINTER(flow_step_t)

    flow_step_vertmax = _libs["divERGe"].get("diverge_flow_step_vertmax", "cdecl")
    flow_step_vertmax.argtypes = [POINTER(flow_step_t), c_voidp]
    flow_step_vertmax.restype = None

    flow_step_loopmax = _libs["divERGe"].get("diverge_flow_step_loopmax", "cdecl")
    flow_step_loopmax.argtypes = [POINTER(flow_step_t), c_voidp]
    flow_step_loopmax.restype = None

    flow_step_chanmax = _libs["divERGe"].get("diverge_flow_step_chanmax", "cdecl")
    flow_step_chanmax.argtypes = [POINTER(flow_step_t), c_voidp]
    flow_step_chanmax.restype = None

    flow_step_euler = _libs["divERGe"].get("diverge_flow_step_euler", "cdecl")
    flow_step_euler.argtypes = [POINTER(flow_step_t), c_double, c_double]
    flow_step_euler.restype = None

    flow_step_niter = _libs["divERGe"].get("diverge_flow_step_niter", "cdecl")
    flow_step_niter.argtypes = [POINTER(flow_step_t)]
    flow_step_niter.restype = index_t

    flow_step_ntimings = _libs["divERGe"].get("diverge_flow_step_ntimings", "cdecl")
    flow_step_ntimings.argtypes = [POINTER(flow_step_t)]
    flow_step_ntimings.restype = index_t

    flow_step_timing = _libs["divERGe"].get("diverge_flow_step_timing", "cdecl")
    flow_step_timing.argtypes = [POINTER(flow_step_t), index_t]
    flow_step_timing.restype = c_double

    flow_step_timing_descr = _libs["divERGe"].get("diverge_flow_step_timing_descr", "cdecl")
    flow_step_timing_descr.argtypes = [POINTER(flow_step_t), index_t]
    flow_step_timing_descr.restype = c_char_p

    flow_step_lambda = _libs["divERGe"].get("diverge_flow_step_lambda", "cdecl")
    flow_step_lambda.argtypes = [POINTER(flow_step_t)]
    flow_step_lambda.restype = c_double

    flow_step_dlambda = _libs["divERGe"].get("diverge_flow_step_dlambda", "cdecl")
    flow_step_dlambda.argtypes = [POINTER(flow_step_t)]
    flow_step_dlambda.restype = c_double

    flow_step_free = _libs["divERGe"].get("diverge_flow_step_free", "cdecl")
    flow_step_free.argtypes = [POINTER(flow_step_t)]
    flow_step_free.restype = None

    class struct_flow_step_vertex_t(Structure):
        pass
    flow_step_vertex_t = struct_flow_step_vertex_t
    struct_flow_step_vertex_t.__slots__ = [
        'ary',
        'q_0',
        'q_1',
        'nk',
        'n_orbff',
        'n_spin',
        'backend',
        'channel'
    ]
    struct_flow_step_vertex_t._fields_ = [
        ('ary', POINTER(complex128_t)),
        ('q_0', index_t),
        ('q_1', index_t),
        ('nk', index_t),
        ('n_orbff', index_t),
        ('n_spin', index_t),
        ('backend', c_char),
        ('channel', c_char)
    ]
    flow_step_vertex = _libs["divERGe"].get("diverge_flow_step_vertex")
    flow_step_vertex.argtypes = [POINTER(flow_step_t), c_char]
    flow_step_vertex.restype = struct_flow_step_vertex_t

    flow_step_refill = _libs["divERGe"].get("diverge_flow_step_refill")
    flow_step_refill.argtypes = [POINTER(flow_step_t), c_double, c_void_p]
    flow_step_refill.restype = None
    flow_step_refill_Hself = _libs["divERGe"].get("diverge_flow_step_refill_Hself")
    flow_step_refill_Hself.argtypes = [POINTER(flow_step_t), c_double, c_void_p]
    flow_step_refill_Hself.restype = c_double


    # Euler integrator
    class struct_euler_t(Structure):
        pass
    euler_t = struct_euler_t
    struct_euler_t.__slots__ = [
        'Lambda',
        'dLambda',
        'Lambda_min',
        'dLambda_min',
        'dLambda_fac',
        'dLambda_fac_scale',
        'maxvert',
        'maxvert_hard_limit',
        'niter',
        'maxiter',
        'consider_maxvert_iter_start',
        'consider_maxvert_lambda'
    ]
    struct_euler_t._fields_ = [
        ('Lambda', c_double),
        ('dLambda', c_double),
        ('Lambda_min', c_double),
        ('dLambda_min', c_double),
        ('dLambda_fac', c_double),
        ('dLambda_fac_scale', c_double),
        ('maxvert', c_double),
        ('maxvert_hard_limit', c_double),
        ('niter', index_t),
        ('maxiter', index_t),
        ('consider_maxvert_iter_start', index_t),
        ('consider_maxvert_lambda', c_double)
    ]

    euler_defaults_CPP = _libs["divERGe"].get("diverge_euler_defaults_CPP", "cdecl")
    euler_defaults_CPP.argtypes = []
    euler_defaults_CPP.restype = euler_t

    euler_next = _libs["divERGe"].get("diverge_euler_next", "cdecl")
    euler_next.argtypes = [POINTER(euler_t), c_double]
    euler_next.restype = c_bool

    # post processing
    class struct_postprocess_conf_t(Structure):
        pass
    struct_postprocess_conf_t.__slots__ = [
        'patch_q_matrices',
        'patch_q_matrices_use_dV',
        'patch_q_matrices_nv',
        'patch_q_matrices_max_rel',
        'patch_q_matrices_eigen_which',
        'patch_V',
        'patch_dV',
        'patch_Lp',
        'patch_Lm',
        'grid_lingap_vertex_file_P',
        'grid_lingap_vertex_file_C',
        'grid_lingap_vertex_file_D',
        'grid_n_singular_values',
        'grid_use_loop',
        'grid_vertex_file',
        'grid_vertex_chan',
        'tu_which_solver_mode',
        'tu_skip_channel_calc',
        'tu_storing_threshold',
        'tu_storing_relative',
        'tu_n_singular_values',
        'tu_lingap',
        'tu_susceptibilities_full',
        'tu_susceptibilities_ff',
        'tu_selfenergy',
        'tu_channels',
        'tu_symmetry_maps',
    ]
    struct_postprocess_conf_t._fields_ = [
        ('patch_q_matrices', c_bool),
        ('patch_q_matrices_use_dV', c_bool),
        ('patch_q_matrices_nv', c_int),
        ('patch_q_matrices_max_rel', c_double),
        ('patch_q_matrices_eigen_which', c_char),
        ('patch_V', c_bool),
        ('patch_dV', c_bool),
        ('patch_Lp', c_bool),
        ('patch_Lm', c_bool),
        ('grid_lingap_vertex_file_P', c_char * int(1024)),
        ('grid_lingap_vertex_file_C', c_char * int(1024)),
        ('grid_lingap_vertex_file_D', c_char * int(1024)),
        ('grid_n_singular_values', c_int),
        ('grid_use_loop', c_bool),
        ('grid_vertex_file', c_char * int(1024)),
        ('grid_vertex_chan', c_char),
        ('tu_which_solver_mode', c_char),
        ('tu_skip_channel_calc', c_bool),
        ('tu_storing_threshold', c_double),
        ('tu_storing_relative', c_double),
        ('tu_n_singular_values', index_t),
        ('tu_lingap', c_bool),
        ('tu_susceptibilities_full', c_bool),
        ('tu_susceptibilities_ff', c_bool),
        ('tu_selfenergy', c_bool),
        ('tu_channels', c_bool),
        ('tu_symmetry_maps', c_bool),
    ]
    postprocess_conf_t = struct_postprocess_conf_t

    postprocess_conf_defaults_CPP = _libs["divERGe"].get("diverge_postprocess_conf_defaults_CPP", "cdecl")
    postprocess_conf_defaults_CPP.argtypes = []
    postprocess_conf_defaults_CPP.restype = postprocess_conf_t

    postprocess_and_write = _libs["divERGe"].get("diverge_postprocess_and_write", "cdecl")
    postprocess_and_write.argtypes = [POINTER(flow_step_t), String]
    postprocess_and_write.restype = None

    postprocess_and_write_finegrained = _libs["divERGe"].get("diverge_postprocess_and_write_finegrained", "cdecl")
    postprocess_and_write_finegrained.argtypes = [POINTER(flow_step_t), String, POINTER(postprocess_conf_t)]
    postprocess_and_write_finegrained.restype = None

    # Function: postprocess_and_write_PY
    # wraps <diverge_postprocess_and_write_finegrained at
    # diverge_postprocess_conf_t.diverge_postprocess_and_write_finegrained> in a
    # convenient pythonic way.
    #
    # Parameters:
    # step - <diverge_flow_step_t> pointer
    # filename - string: where to save the postprocessing results
    # **kwargs - each member of <diverge_postprocess_conf_t> can be passed as
    #            kwarg (key1 = val1, key2 = val2, …). The defaults are those
    #            returned by <diverge_postprocess_conf_defaults_CPP at
    #            diverge_postprocess_conf_t.diverge_postprocess_conf_defaults_CPP>.
    def postprocess_and_write_PY( step, filename, **kwargs ):
        cfg = postprocess_conf_defaults_CPP()
        for s in struct_postprocess_conf_t.__slots__:
            try:
                v = kwargs.pop(s)
                cfg.__setattr__( s, v )
            except KeyError:
                pass
        for k in kwargs.keys():
            mpi_py_eprint( "postprocess_and_write_PY: unused kwarg '%s'" % k )
        return postprocess_and_write_finegrained( step, filename, byref(cfg) )

    # Model output
    model_output_set_npath = _libs["divERGe"].get("diverge_model_output_set_npath", "cdecl")
    model_output_set_npath.argtypes = [c_int]
    model_output_set_npath.restype = None

    model_to_file = _libs["divERGe"].get("diverge_model_to_file", "cdecl")
    model_to_file.argtypes = [POINTER(model_t), String]
    if sizeof(c_int) == sizeof(c_void_p):
        model_to_file.restype = ReturnString
    else:
        model_to_file.restype = String
        model_to_file.errcheck = ReturnString

    # config
    class struct_model_output_conf_t(Structure):
        pass
    struct_model_output_conf_t.__slots__ = [
        'kc', 'kf', 'kc_ibz_path', 'kf_ibz_path', 'H', 'U', 'E', 'npath', 'fatbands',
    ]
    struct_model_output_conf_t._fields_ = [
        ('kc', c_int),
        ('kf', c_int),
        ('kc_ibz_path', c_int),
        ('kf_ibz_path', c_int),
        ('H', c_int),
        ('U', c_int),
        ('E', c_int),
        ('npath', c_int),
        ('fatbands', c_int),
    ]
    model_output_conf_t = struct_model_output_conf_t

    model_output_conf_defaults_CPP = _libs["divERGe"].get("diverge_model_output_conf_defaults_CPP", "cdecl")
    model_output_conf_defaults_CPP.argtypes = []
    model_output_conf_defaults_CPP.restype = model_output_conf_t

    model_to_file_finegrained = _libs["divERGe"].get("diverge_model_to_file_finegrained", "cdecl")
    model_to_file_finegrained.argtypes = [POINTER(model_t), String, POINTER(model_output_conf_t)]
    if sizeof(c_int) == sizeof(c_void_p):
        model_to_file_finegrained.restype = ReturnString
    else:
        model_to_file_finegrained.restype = String
        model_to_file_finegrained.errcheck = ReturnString

    # Function: model_to_file_PY
    # writes a <diverge_model_t> to file using the
    # <diverge_model_to_file_finegrained at
    # diverge_model_output_conf_t.diverge_model_to_file_finegrained> backend
    # wrapped in a convenient pythonic way. Returns the md5sum of the model file
    # as string.
    #
    # Parameters:
    # model - <diverge_model_t> pointer
    # filename - string: where to save the model
    # **kwargs - each member of <diverge_model_output_conf_t> can be passed as
    #            kwarg (key1 = val1, key2 = val2, …). The defaults are those
    #            returned by <diverge_model_output_conf_defaults_CPP at
    #            diverge_model_output_conf_t.diverge_model_output_conf_defaults_CPP>.
    def model_to_file_PY( model, filename, **kwargs ):
        cfg = model_output_conf_defaults_CPP()
        for s in struct_model_output_conf_t.__slots__:
            try:
                v = kwargs.pop(s)
                cfg.__setattr__( s, v )
            except KeyError:
                pass
        for k in kwargs.keys():
            mpi_py_eprint( "model_to_file_PY: unused kwarg '%s'" % k )
        return model_to_file_finegrained( model, filename, byref(cfg) )

    # Momentum generator
    model_generate_meshes = _libs["divERGe"].get("diverge_model_generate_meshes", "cdecl")
    model_generate_meshes.argtypes = [POINTER(c_double), POINTER(c_double), index_t * int(3), index_t * int(3), (c_double * int(3)) * int(3)]
    model_generate_meshes.restype = None

    model_generate_mom_basis = _libs["divERGe"].get("diverge_model_generate_mom_basis", "cdecl")
    model_generate_mom_basis.argtypes = [(c_double * int(3)) * int(3), (c_double * int(3)) * int(3)]
    model_generate_mom_basis.restype = None

    # Patching
    patching_find_fs_pts_C_ = _libs["divERGe"].get("diverge_patching_find_fs_pts_C", "cdecl")
    patching_find_fs_pts_C_.argtypes = [POINTER(model_t), POINTER(c_double), index_t, index_t, index_t, POINTER(POINTER(index_t)), POINTER(index_t)]
    patching_find_fs_pts_C_.restype = None
    # Function: patching_find_fs_pts_PY
    # returns a numpy array of <index_t> with the indices of the found patches
    #
    # Parameters:
    # model - <diverge_model_t> pointer
    # energies - either None or pointer to a double (64bit) floating point array
    #            that holds energies for nbands bands on the mesh defined in the
    #            model, i.e. an (nk, nb) array
    # nbands - number of bands to use. if energies is None, must coincide with the
    #          number of bands defined in the model
    # np_ibz - number of patches that should be found in the IBZ
    # np_ibz_search - number of points that are considered for the patch search
    def patching_find_fs_pts_PY( model, energies, nbands, np_ibz, np_ibz_search ):
        if energies is None:
            e_ptr = None
        else:
            e_ptr = energies.ctypes.data
        pts = (POINTER(index_t))()
        npts = index_t()
        patching_find_fs_pts_C_( model, e_ptr, nbands, np_ibz,
                    np_ibz_search, ctypes.byref(pts), ctypes.byref(npts) )
        pts_ary = view_array( pts, dtype=index_t, shape=(np.array(npts),) )
        pts_cpy = np.copy( pts_ary )
        mem_free( pts )
        return pts_cpy

    patching_free = _libs["divERGe"].get("diverge_patching_free", "cdecl")
    patching_free.argtypes = [POINTER(mom_patching_t)]
    patching_free.restype = None

    patching_from_indices = _libs["divERGe"].get("diverge_patching_from_indices", "cdecl")
    patching_from_indices.argtypes = [POINTER(model_t), c_voidp, index_t]
    patching_from_indices.restype = POINTER(mom_patching_t)

    patching_autofine = _libs["divERGe"].get("diverge_patching_autofine", "cdecl")
    patching_autofine.argtypes = [POINTER(model_t), POINTER(mom_patching_t), POINTER(c_double), index_t, index_t, c_double, c_double, c_double]
    patching_autofine.restype = None

    patching_symmetrize_refinement = _libs["divERGe"].get("diverge_patching_symmetrize_refinement", "cdecl")
    patching_symmetrize_refinement.argtypes = [POINTER(model_t), POINTER(mom_patching_t)]
    patching_symmetrize_refinement.restype = None

    # Symmetrize
    generate_symm_maps = _libs["divERGe"].get("diverge_generate_symm_maps", "cdecl")
    generate_symm_maps.argtypes = [POINTER(model_t)]
    generate_symm_maps.restype = None

    symmetrize_2pt_coarse = _libs["divERGe"].get("diverge_symmetrize_2pt_coarse", "cdecl")
    symmetrize_2pt_coarse.argtypes = [POINTER(model_t), POINTER(complex128_t), POINTER(complex128_t)]
    symmetrize_2pt_coarse.restype = c_double

    symmetrize_2pt_fine = _libs["divERGe"].get("diverge_symmetrize_2pt_fine", "cdecl")
    symmetrize_2pt_fine.argtypes = [POINTER(model_t), POINTER(complex128_t), POINTER(complex128_t)]
    symmetrize_2pt_fine.restype = c_double

    symmetrize_mom_coarse = _libs["divERGe"].get("diverge_symmetrize_mom_coarse", "cdecl")
    symmetrize_mom_coarse.argtypes = [POINTER(model_t), POINTER(c_double), index_t, POINTER(c_double)]
    symmetrize_mom_coarse.restype = c_double

    symmetrize_mom_fine = _libs["divERGe"].get("diverge_symmetrize_mom_fine", "cdecl")
    symmetrize_mom_fine.argtypes = [POINTER(model_t), POINTER(c_double), index_t, POINTER(c_double)]
    symmetrize_mom_fine.restype = c_double

    # Additional symmetry related stuff
    merge_rs_orb = _libs["divERGe"].get("merge_rs_orb", "cdecl")
    merge_rs_orb.argtypes = [POINTER(model_t)]
    merge_rs_orb.restype = None

    # Testing
    run_tests_ = _libs["divERGe"].get("diverge_run_tests", "cdecl")
    run_tests_.argtypes = [c_int, POINTER(POINTER(c_char))]
    run_tests_.restype = c_int
    # Function: run_tests
    # run all unit tests shipped with divERGe to check health.
    #
    # Parameters:
    # args - list of strings to pass as an argument to CATCH (the testing
    #        framework). To enable a specific test filter, set args=['[filter]'].
    #        For example, the BHK tests can be run with args=['[BHK]'].
    def run_tests( args=[] ):
        try:
            arr = (c_char_p * (len(args) + 1))()
            arr[1:] = [ a.encode('utf8') for a in args ]
            arr[0] = b"diverge.py"
            return run_tests_( len(arr), POINTER(POINTER(c_char))(arr) )
        except:
            mpi_py_eprint("could not find test function")
            return 1

    # hacking
    model_hack = _libs["divERGe"].get("diverge_model_hack", "cdecl")
    model_hack.argtypes = [POINTER(model_t), String, String]
    model_hack.restype = None

    model_print_hacks = _libs["divERGe"].get("diverge_model_print_hacks", "cdecl")
    model_print_hacks.argtypes = None
    model_print_hacks.restype = None

    # shared memory...
    shared_malloc = _libs['divERGe'].get('shared_malloc', 'cdecl')
    shared_malloc.argtypes = [c_int64]
    shared_malloc.restype = c_voidp
    shared_calloc = _libs['divERGe'].get('shared_calloc', 'cdecl')
    shared_calloc.argtypes = [c_int64, c_int64]
    shared_calloc.restype = c_voidp
    shared_free = _libs['divERGe'].get('shared_free', 'cdecl')
    shared_free.argtypes = [c_voidp]
    shared_free.restype = None
    shared_exclusive_enter = _libs['divERGe'].get('shared_exclusive_enter', 'cdecl')
    shared_exclusive_enter.argtypes = [c_voidp]
    shared_exclusive_enter.restype = c_int
    shared_exclusive_wait = _libs['divERGe'].get('shared_exclusive_wait', 'cdecl')
    shared_exclusive_wait.argtypes = [c_voidp]
    shared_exclusive_wait.restype = None
    shared_malloc_rank = _libs['divERGe'].get('shared_malloc_rank', 'cdecl')
    shared_malloc_rank.argtypes = []
    shared_malloc_rank.restype = c_int
    shared_malloc_size = _libs['divERGe'].get('shared_malloc_size', 'cdecl')
    shared_malloc_size.argtypes = []
    shared_malloc_size.restype = c_int
    shared_malloc_barrier = _libs['divERGe'].get('shared_malloc_barrier', 'cdecl')
    shared_malloc_barrier.argtypes = []
    shared_malloc_barrier.restype = None

    # redo the typedefs
    rs_hopping_t = struct_rs_hopping_t
    rs_vertex_t = struct_rs_vertex_t
    mom_patching_t = struct_mom_patching_t
    model_t = struct_model_t
    internals_t = struct_internals_t
    tu_formfactor_t = struct_tu_formfactor_t
    flow_step_t = struct_flow_step_t
    euler_t = struct_euler_t

    # Var: numpy
    # imports numpy in standard fashion (as np)
    import numpy as np
    # Var: zeros
    # additionally import zeros from numpy for less namespace pollution
    zeros = np.zeros

    # Function: view_array
    # give an array view on existing memory
    #
    # Parameters:
    #   dtype - data type that the view is using
    #   shape - all dimensions (in C ordering) that the view is using
    def view_array( mem, dtype=np.complex128, shape=(1,) ):
        dtype = np.dtype(dtype)
        return np.ctypeslib.as_array( cast(mem, POINTER(c_char)), shape=(*shape,dtype.itemsize) ).view( dtype=dtype ).reshape( shape )

    # Function: alloc_array
    # allocate an array of given data type and shape such that it is not cleared by
    # the python garbage collector
    #
    # Parameters:
    #   shape - all dimensions (in C ordering)
    #   dtype - data type. can be <complex128_t>, <rs_hopping_t>, <rs_vertex_t>, or
    #           <tu_formfactor_t>. data type name must be passed as python string.
    def alloc_array( shape, dtype = "complex128_t" ):
        if dtype == "complex128_t":
            return view_array( mem_alloc_complex128_t(np.prod(shape)), dtype=np.complex128, shape=shape )
        elif dtype == "rs_hopping_t":
            return view_array( mem_alloc_rs_hopping_t(np.prod(shape)), dtype=struct_rs_hopping_t, shape=shape )
        elif dtype == "rs_vertex_t":
            return view_array( mem_alloc_rs_vertex_t(np.prod(shape)), dtype=struct_rs_vertex_t, shape=shape )
        elif dtype == "tu_formfactor_t":
            return view_array( mem_alloc_tu_formfactor_t(np.prod(shape)), dtype=struct_tu_formfactor_t, shape=shape )
        else:
            mpi_py_eprint("cannot allocate array of type '%s'" % dtype)
            return None

    print_ = print
    # Function: print
    # wraps the python print function to work with MPI
    def print( *args, **kwargs ):
        if mpi_comm_rank() == 0:
            print_( *args, **kwargs )

    # Function: autoflow
    # simplest possible FRG flow
    #
    # Parameters:
    #   hoppings - numpy array of dtype <rs_hopping_t>, *required*.
    #   vertex - numpy array of dtype <rs_vertex_t>, *required*.
    #   nk - number of coarse k points in all three dimensions, *required*.
    #        Dimensionality of the model is inferred from the nk list by inspecting
    #        how many nonzero entries exist.
    #   nkf - number of fine k points in all three dimensions, defaults to 15 in
    #         each dimension for mode=="grid" or mode=="tu", and to 1 for
    #         mode=="patch".
    #   lattice - 3×3 matrix in C order that holds the lattice vectors lattice[0],
    #             lattice[1], and lattice[2]
    #   no - number of orbitals and sites
    #   SU2 - SU2 symmetry setting. If SU2>0, model is assumed to be SU2 symmetric.
    #         If SU2==0, model is not SU2 symmetric and exchange symmetry in the
    #         vertex is *not* enforced. for SU2<0, model is not SU2 symmetric, but
    #         exchange symmetry in the vertex *is* enforced.
    #   nspin - number of spin indices. For SU2>0, nspin=1 is required. For SU2<=0,
    #           nspin must be set from physical grounds.
    #   sites - positions of the orbital centers in real space as array of shape
    #           (no, 3).
    #   model_output - output file for the model.
    #   post_output - output file for postprocessing info.
    #   flow_output - output file for flow data.
    #   rs_symmetries - realspace symmetry list as array of shape (nsym, 3, 3). If
    #                   None, no symmetries are used.
    #   orb_symmetries - orbital symmetry list as array of shape (nsym, no, no). If
    #                    None, no symmetries are used.
    #   mode - backend selection. valid choices are "grid", "patch", and "tu".
    #   npatches_ibz - number of FS patches in the IBZ used with the "patch"
    #                  backend.
    #   formfactor_maxdist - maximal distance up to which formfactors should be
    #                        included in the "tu" backend.
    #   channels - selection of channels to be included in the FRG flow. The library
    #              looks for the characters 'P', 'C', 'D', 'S' in the string that
    #              stand for the particle particle channel ('P'), the crossed
    #              particle hole channel ('C'), the direct particle hole channel
    #              ('D'), and self-energies ('S'), respectively.
    #   ibz_path - Brillouin zone path in reciprocal crystal coordinates. if empty,
    #              do not compute band structure
    #   maxvert - value after which the vertex is considered 'diverged'
    #   mu - chemical potential. use if mu is not None
    #   nu - filling value. use if nu is not None. nu has precedence over mu, i.e.
    #        if both nu and mu are not None, the filling value is used.
    #
    # Create the (numpy) arrays with
    # === Python ===
    # import diverge as dvg
    # hoppings = dvg.zeros(shape_h, dtype=dvg.rs_hopping_t)
    # vertex = dvg.zeros(shape_v, dtype=dvg.rs_vertex_t)
    # ==============
    def autoflow( hoppings=None, vertex=None, nk=None, nkf=None, lattice=np.eye(3),
                 no=1, SU2=1, nspin=1, sites=[[0,0,0]], model_output="model.dvg",
                 post_output="post.dvg", flow_output="flow.dat", rs_symmetries=None,
                 orb_symmetries=None, mode="tu", npatches_ibz=6,
                 formfactor_maxdist=2.01, channels="PCD", ibz_path=[], maxvert=50.0,
                 mu=None, nu=None ):

        errors = 0
        init(None, None)
        compilation_status()

        if (hoppings is None) or (vertex is None) or (nk is None):
            mpi_py_eprint("must provide hoppings, vertex, and nk")
            errors += 1

        if len(sites) != no:
            mpi_py_eprint("sites array must be of shape (%i, 3)" % no)
            errors += 1

        if (not (rs_symmetries is None)) and (not (orb_symmetries is None)):
            num_sym = len(rs_symmetries)
            if num_sym != len(orb_symmetries):
                mpi_py_eprint("number or orbital symmetries does not match number of realspace symmetries")
                errors += 1
            if rs_symmetries.shape != (num_sym, 3, 3):
                mpi_py_eprint("realspace symmetries must be of shape (%i, 3, 3)" % num_sym)
                errors += 1
            if orb_symmetries.shape != (num_sym, no, no):
                mpi_py_eprint("orbital symmetries must be of shape (%i, %i, %i)" % (num_sym, no, no))
                errors += 1

        if errors > 0:
            mpi_py_eprint("aborting due to previous errors")
            mpi_exit(errors)

        if SU2>0 and nspin != 1:
            nspin = 1
            mpi_py_eprint("for SU2>0, nspin=1 must be set; resetting")

        if nkf is None:
            nkf = np.copy(nk)
            default_fac = 1 if mode == "patch" else 15
            nkf[nkf != 0] = default_fac
            mpi_py_eprint("using refinement of factor %i as default" % default_fac)

        mean_hop = np.abs(np.array(hoppings['t']).view(np.complex128)).sum()
        mean_vert = np.abs(np.array(vertex['V']).view(np.complex128)).sum()
        if mean_hop < 0.5 or mean_hop > 50:
            mpi_py_eprint("consider readjusting your units, autoflow is optimized for hoppings of order one.")
        if mean_vert < 0.5 or mean_vert > 50:
            mpi_py_eprint("consider readjusting your units, autoflow is optimized for vertices of order one.")

        model = model_init()

        model.contents.name = b"autoflow"

        model.contents.nk[:] = nk
        model.contents.nkf[:] = nkf

        lattice_view = view_array( model.contents.lattice, dtype=np.float64, shape=(3,3) )
        lattice_view[:,:] = lattice

        model.contents.n_orb = no
        positions_view = view_array( model.contents.positions, dtype=np.float64, shape=(no,3) )
        positions_view[:,:] = sites

        model.contents.SU2 = SU2
        model.contents.n_spin = nspin

        hoppings_copy = alloc_array( hoppings.shape, "rs_hopping_t" )
        hoppings_copy[:] = hoppings
        model.contents.hop = hoppings_copy.ctypes.data
        model.contents.n_hop = hoppings_copy.size

        vertex_copy = alloc_array( vertex.shape, "rs_vertex_t" )
        vertex_copy[:] = vertex
        model.contents.vert = vertex_copy.ctypes.data
        model.contents.n_vert = vertex_copy.size

        if not orb_symmetries is None and not rs_symmetries is None:
            mpi_py_eprint( "setting symmetries" )
            orbsym_copy = alloc_array( orb_symmetries.shape, "complex128_t" )
            orbsym_copy[:] = orb_symmetries
            model.contents.orb_symmetries = orbsym_copy.ctypes.data
            rssym_view = view_array( model.contents.rs_symmetries, dtype=np.float64, shape=rs_symmetries.shape )
            rssym_view[:] = rs_symmetries
            model.contents.n_sym = orbsym_copy.shape[0]

        if len(ibz_path) > 0:
            model.contents.n_ibz_path = len(ibz_path)
            ibz_view = view_array( model.contents.ibz_path, dtype=np.float64, shape=(len(ibz_path), 3) )
            ibz_view[:] = ibz_path

        validate = model_validate( model )
        if validate:
            mpi_py_eprint("invalid model")
            mpi_exit( validate )

        model_internals_common( model )

        set_fill = False
        if not mu is None:
            model_set_chempot( model, None, -1, mu )
            set_fill = True
        if not nu is None:
            model_set_filling( model, None, -1, nu )
            set_fill = True
        if set_fill:
            mpi_py_eprint( "adjusted mu/nu. filling value: %.3f" % model_get_filling( model, None, -1 ) )

        if mode == "grid":
            model_internals_grid( model )
        elif mode == "patch":
            model_internals_patch( model, npatches_ibz )
        elif mode == "tu":
            model_internals_tu( model, formfactor_maxdist )
        checksum = model_to_file( model, model_output )
        mpi_py_eprint( "model output to %s, checksum %s" % (model_output, checksum) )

        step = flow_step_init( model, mode, channels )
        maxs = dict(vert=np.zeros(1), loop=np.zeros(2), chan=np.zeros(3))
        eu = euler_defaults_CPP()
        eu.dLambda_fac_scale = 1.0
        eu.dLambda_fac = 0.1
        eu.maxvert = maxvert

        mpi_py_eprint( "flow output to %s…" % flow_output )
        stdout_ = sys.stdout
        sys.stdout = mpi_stdout_logger( flow_output )

        print( "# %9s %12s %11s %11s %11s %11s %11s %11s" % ('Lambda', 'dLambda', 'Lp', 'Lm', 'dP', 'dC', 'dD', 'V') )
        while True:
            flow_step_euler( step, eu.Lambda, eu.dLambda )
            flow_step_vertmax( step, maxs['vert'].ctypes.data )
            flow_step_loopmax( step, maxs['loop'].ctypes.data )
            flow_step_chanmax( step, maxs['chan'].ctypes.data )
            print( "%.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e" %
                  (eu.Lambda, eu.dLambda, *maxs['loop'], *maxs['chan'], *maxs['vert']) )
            eu_next = euler_next( byref(eu), maxs['vert'][0] )
            if not eu_next:
                break
            sys.stdout.flush()

        sys.stdout = stdout_

        mpi_py_eprint( "postprocessing output to %s…" % post_output )
        postprocess_and_write( step, post_output )

        flow_step_free( step )
        model_free( model )
        finalize()

    # Class: mpi_stdout_logger
    # direct output to stdout and a file at the same time. usage:
    # === Python ===
    # stdout_ = sys.stdout
    # sys.stdout = mpi_stdout_logger(filename)
    # # some code with print() statements
    # sys.stdout = stdout_
    # ==============
    class mpi_stdout_logger(object):
        def __init__(self, fname, mode="w"):
            if mpi_comm_rank() == 0:
                self.terminal = sys.stdout
                self.log = open(fname, mode)
        def write(self, message):
            if mpi_comm_rank() == 0:
                self.terminal.write(message)
                self.log.write(message)
        def flush(self):
            if mpi_comm_rank() == 0:
                self.terminal.flush()
                self.log.flush()

else: # _libs["divERGe"] is None
    print( info() )

# Namespace: Structured Arrays
# DivERGe defines several primitive datatypes (structures) that can be
# interfaced to from this python library:
#
# - <complex128_t>
# - <rs_hopping_t>
# - <rs_vertex_t>
# - <tu_formfactor_t>
# - <sym_op_t>
# - <site_descr_t>
#
# Allocation:
# To allocate an array that is *not* garbage collected from python (and can thus
# be free'd from within divERGe or via <diverge.mem_free at diverge_mem_free>),
# call <diverge.alloc_array at dvg.alloc_array>. The datatype is passed as
# *string* to this function, in contrast to what one may be used to from numpy.
# For standard, garbage-collected allocation, use <diverge.zeros at dvg.zeros>
# (or just np.zeros):
# === Python ===
# hopping_gc = diverge.zeros( (128,), dtype=diverge.rs_hopping_t )
# hopping_non_gc = diverge.alloc_array( (128,), dtype="rs_hopping_t" )
# ==============
#
# View:
# A numpy structured array can be used just as any other numpy array (shapes,
# views, etc). To conveniently use memory owned by divERGe in a pythonic way, we
# added <diverge.view_array at dvg.view_array>:
# === Python ===
# positions = diverge.view_array( model.contents.positions, dtype=np.float64, shape=(12,3) )
# positions[:] += 1 # now positions are updated!
# ==============
#
# Access:
# Now to the usage of structured arrays (or array views). If one or more full
# element(s) is (are) to be set, a (list of) tuple(s) can be passed as so:
# === Python ===
# hopping = diverge.alloc_array( (128,), dtype="rs_hopping_t" )
# hopping[0] = ( [0,0,0], 0,0, 0,0, (1.0,0.0) )
# hopping[1:3] = [ ( [0,0,0], 1,1, 0,0, (0.2,0.0) ),
#                  ( [0,0,0], 2,2, 0,0, (0.2,0.0) ) ]
# ==============
# This works because the inserted tuples are understood as compatible with the
# <rs_hopping_t> data type:
# ---- C/C++ ----
# // C declaration
# struct rs_hopping_t {
#   index_t R[3];
#   index_t o1,o2, s1,s2;
#   complex128_t t;
# };
# ---------------
# If only a single member of the structure should be modified or accessed, a
# dictionary-like syntax is supported:
# === Python ===
# vertex = diverge.alloc_array( (128,), dtype="rs_vertex_t" )
# vertex[0]['chan'] = 'D'
# vertex[0]['V'] = (3.0,0.0)
# ==============
# The keys of this dictionary-like access resemble the names of the members in
# the respective C declaration.

