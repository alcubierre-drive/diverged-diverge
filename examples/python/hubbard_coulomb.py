#!/usr/bin/env python3
#coding:utf8

import diverge
import numpy as np

USE_GRID = False

def coulomb(x,y,U):
    d = np.sqrt(x**2+y**2)
    if(d < 0.5):
        return U
    elif(d < 10.0):
        return U/d
    else:
        return 0.0

# initialization. takes the C-style argc, argv as arguments, which can be safely
# ignored (i.e. set to None) in most cases.
diverge.init(None, None)
# we dont need all the verbosity, thus we set the loglevel to 1
diverge.mpi_loglevel_set(1)
diverge.compilation_status()
# save the MPI rank for later. if you don't know what this means, ignore.
myrank = diverge.mpi_comm_rank()

################################################################################
############################## MODEL SETUP #####################################
################################################################################
# initialize model
model = diverge.model_init()

# set model name (not required)
model.contents.name = b'Python Hubbard Model'

# set number of k points. by default, all nk are zero. to simulate a two
# dimensional model, we must leave the last length zero.
model.contents.nk[0] = 8 if USE_GRID else 256
model.contents.nk[1] = 8 if USE_GRID else 256
model.contents.nkf[0] = 15 if USE_GRID else 1
model.contents.nkf[1] = 15 if USE_GRID else 1

# set the realspace lattice vectors. again, by default all are zero, i.e. we
# only need to set the indices that are nonzero. For numpy-like behavior we use
# diverge.view_array
lattice = diverge.view_array( model.contents.lattice, dtype=np.float64, shape=(3,3) )
lattice[:,:] = np.eye(3)

# number of orbitals/sites
model.contents.n_orb = 1

### HOPPING PARAMETERS BEGIN ###
# allocate hopping array. we *must use alloc_array from diverge. because the memory
# behind the array is assumed to be owned by the library
hoppings = diverge.alloc_array( (9,), "rs_hopping_t" )
tp = -0.2
hoppings[0] = ([ 0,0,0], 0,0,0,0, (4*tp,0.0)) # chemical potential
hoppings[1] = ([ 1,0,0], 0,0,0,0, (1.0,0.0))
hoppings[2] = ([-1,0,0], 0,0,0,0, (1.0,0.0))
hoppings[3] = ([0, 1,0], 0,0,0,0, (1.0,0.0))
hoppings[4] = ([0,-1,0], 0,0,0,0, (1.0,0.0))
hoppings[5] = ([ 1, 1,0], 0,0,0,0, (tp,0.0))
hoppings[6] = ([-1, 1,0], 0,0,0,0, (tp,0.0))
hoppings[7] = ([ 1,-1,0], 0,0,0,0, (tp,0.0))
hoppings[8] = ([-1,-1,0], 0,0,0,0, (tp,0.0))
# set hoppings in model
model.contents.hop = hoppings.ctypes.data
model.contents.n_hop = hoppings.size
### HOPPING PARAMETERS END ###

model.contents.SU2 = True
model.contents.n_spin = 1

### SYMMETRIES BEGIN ###
nsym = 8
model.contents.n_sym = nsym

# allocate orb_symmetries array. again, we *must use alloc_array*.
orb_symmetries = diverge.alloc_array( (nsym,), dtype="complex128_t" )
# set orb_symmetries, they are trivial.
orb_symmetries[:] = 1.0
# set orb_symmetries in model
model.contents.orb_symmetries = orb_symmetries.ctypes.data

# give array view on rs_symmetries. since rs_symmetries is stack memory it is
# already allocated. it is again advised to *use view_array from diverge.
rs_symmetries = diverge.view_array( model.contents.rs_symmetries, dtype=np.float64, shape=(nsym,3,3) )
# set symmetry content. We use C4v here.
rotation = lambda theta: np.array([[ np.cos(theta), np.sin(theta), 0],
                                   [-np.sin(theta), np.cos(theta), 0],
                                   [             0,             0, 1]])
def reflection( axis_x ):
    r = np.eye(3)
    if axis_x:
        r[0,0] = -1
    else:
        r[1,1] = -1
    return r
rs_symmetries[0] = rotation(0*np.pi/2)
rs_symmetries[1] = rotation(1*np.pi/2)
rs_symmetries[2] = rotation(2*np.pi/2)
rs_symmetries[3] = rotation(3*np.pi/2)
rs_symmetries[4] = reflection(True)
rs_symmetries[5] = reflection(False)
rs_symmetries[6] = rotation( np.pi/4) @ reflection(True) @ rotation(-np.pi/4)
rs_symmetries[7] = rotation(-np.pi/4) @ reflection(True) @ rotation( np.pi/4)
### SYMMETRIES END ###

# allocate vertex object. same story, *use alloc_array*
vertex = diverge.alloc_array( (25*25,), dtype="rs_vertex_t" )
for x in range(-12,13):
    for y in range(-12,13):
        vertex[y + 25 * x] = ('D', [x,y,0], 0,0,0,0,0,0, (coulomb(x,y,3.0),0.0));
# set vertex in model
model.contents.vert = vertex.ctypes.data
model.contents.n_vert = vertex.size

# set the IBZ path in reciprocal crystal coordinates (i.e. in units of
# reciprocal lattice vectors). again the default value for the
# model.contents.ibz_path array is zero, so we only set what we need. for
# convenience and numpy-like behavior we use view_array()
ibz_path = diverge.view_array( model.contents.ibz_path, dtype=np.float64, shape=(4,3) )
ibz_path[1,:] = [0.5,0,0]
ibz_path[2,:] = [0.5,0.5,0]
model.contents.n_ibz_path = ibz_path.shape[0]

################################################################################
###################### MODEL VALIDATION AND INTERNALS ##########################
################################################################################

# validate model
if diverge.model_validate( model ):
    # we use the diverge.print function because we want to only see the output once,
    # in case we are mpi rank 0.
    diverge.print("invalid model")

# and initialize internal structures
diverge.model_internals_common( model )

if USE_GRID:
    # initialize internal grid structures
    diverge.model_internals_grid( model )
else:
    # initialize internal patch structures
    diverge.model_internals_patch( model, 6 )

# now write the model output including internals (True)
checksum = diverge.model_to_file( model, "mod.dvg", True )

################################################################################
######################### FLOW STEPPER AND FLOW ################################
################################################################################

# initialize flow step
step = diverge.flow_step_init( model, "grid" if USE_GRID else "patch", "PCD" )

# some auxiliary variables and the custom euler stepper (eu)
maxs = dict(vert=np.zeros(1), loop=np.zeros(2), chan=np.zeros(3))
eu = diverge.euler_defaults_CPP()
eu.dLambda_fac_scale = 1.0
eu.vertmax = 2000
eu.dLambda_fac = 0.1

# and finally, an FRG flow!
diverge.print( "Lambda dLambda Lp Lm dP dC dD V" )
while True:
    # this function performs the flow step from Lambda to Lambda+dLambda
    diverge.flow_step_euler( step, eu.Lambda, eu.dLambda )
    # and here are some functions that provide useful output
    diverge.flow_step_vertmax( step, maxs['vert'].ctypes.data )
    diverge.flow_step_loopmax( step, maxs['loop'].ctypes.data )
    diverge.flow_step_chanmax( step, maxs['chan'].ctypes.data )
    # which we also want to print
    diverge.print( "%.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e" % (eu.Lambda, eu.dLambda, *maxs['loop'], *maxs['chan'], *maxs['vert']) )
    # we advance the stepper which automatically sets dLambda and Lambda
    eu_next = diverge.euler_next( diverge.byref(eu), maxs['vert'][0] )
    if not eu_next:
        break

################################################################################
############################### POST PROCESSING ################################
################################################################################

# write the flow output with default settings (None)
diverge.postprocess_and_write( step, None, "out.dvg" )

# clean up all the divERGe internals
diverge.flow_step_free( step )
diverge.model_free( model )
diverge.finalize()
