#!/usr/bin/env python3
#coding:utf8

import diverge

# diverge.zeros == np.zeros
hoppings = diverge.zeros( 9, dtype=diverge.rs_hopping_t )
vertex = diverge.zeros( 1, dtype=diverge.rs_vertex_t )

tp = -0.27
# hopping format: ( [Rx,Ry,Rz], o1,o2, s1,s2, (t_real,t_imag) )
# chemical potential
hoppings[0] = ([ 0,0,0], 0,0,0,0, (4*tp,0.0))
# nearest neighbor hoppings
hoppings[1] = ([ 1,0,0], 0,0,0,0, (1.0,0.0))
hoppings[2] = ([-1,0,0], 0,0,0,0, (1.0,0.0))
hoppings[3] = ([0, 1,0], 0,0,0,0, (1.0,0.0))
hoppings[4] = ([0,-1,0], 0,0,0,0, (1.0,0.0))
# next nearest neighbor hoppings (diagonal)
hoppings[5] = ([ 1, 1,0], 0,0,0,0, (tp,0.0))
hoppings[6] = ([-1, 1,0], 0,0,0,0, (tp,0.0))
hoppings[7] = ([ 1,-1,0], 0,0,0,0, (tp,0.0))
hoppings[8] = ([-1,-1,0], 0,0,0,0, (tp,0.0))

# vertex format: ( channel, [Rx,Ry,Rz], o1,o2, s1,s2,s3,s4, (V_real,V_imag) )
# Hubbard U
vertex[0] = ('D', [0,0,0], 0,0, 0,0,0,0, (3.0,0.0))


diverge.autoflow( hoppings=hoppings, vertex=vertex, nk=[20,20,0] )
# autoflow defaults to TUFRG. To call an N-patch FRG calculation (symmetries
# recommended, default FS patching is fragile), use
#
# diverge.autoflow( hoppings=hoppings, vertex=vertex, nk=[800,800,0], mode="patch", npatches_ibz=32 )
#
# To call a grid FRG calculation instead use
#
# diverge.autoflow( hoppings=hoppings, vertex=vertex, nk=[8,8,0], mode="grid" )
