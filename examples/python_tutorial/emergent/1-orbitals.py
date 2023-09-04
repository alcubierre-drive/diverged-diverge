#!/usr/bin/env python3
#coding:utf8
import diverge
import numpy as np

# some parameters
params_emery = dict( Ud=3.0, Up=1.0, tdd=0.2, tpp=0.3, tdp=1.0 )

positions = np.array( [[0,0,0], [0.5,0,0], [0,0.5,0]] )
lattice = np.eye(3)

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

# fill the hop_list and generate hop_ary
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
hop_array = diverge.zeros( (len(hop_list),), dtype=diverge.rs_hopping_t )
hop_array[:] = hop_list
print( "using %i hopping elements" % len(hop_list) )

# we have a Hubbard U in all of the orbitals!
vertices = diverge.zeros( (3,), dtype=diverge.rs_vertex_t )
vertices[0] = ( 'D', [0,0,0], 0,0, -1,0,0,0, (params_emery['Ud'],0.0) )
vertices[1] = ( 'D', [0,0,0], 1,1, -1,0,0,0, (params_emery['Up'],0.0) )
vertices[2] = ( 'D', [0,0,0], 2,2, -1,0,0,0, (params_emery['Up'],0.0) )

# we want a band structure -> we need the irreducible path
ibz_path = np.array( [ [0,0,0], [0,0.5,0], [0.5,0.5,0], [0,0,0] ] )

# let it flow
diverge.autoflow( hop_array, vertices, nk=[16,16,0], nkf=[5,5,0], no=3,
                  lattice=lattice, sites=positions, ibz_path=ibz_path )

