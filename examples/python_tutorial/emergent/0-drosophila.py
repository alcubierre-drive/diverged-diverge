#!/usr/bin/env python3
#coding:utf8
import diverge

hoppings = diverge.zeros( (4,), dtype=diverge.rs_hopping_t )
# ( [Rx,Ry,Rz], o1,o2, s1,s2, (t_real,t_imag) )
hoppings[0] = ( [0, 1,0], 0,0, 0,0, (-1.0,0.0) )
hoppings[1] = ( [0,-1,0], 0,0, 0,0, (-1.0,0.0) )
hoppings[2] = ( [ 1,0,0], 0,0, 0,0, (-1.0,0.0) )
hoppings[3] = ( [-1,0,0], 0,0, 0,0, (-1.0,0.0) )

vertices = diverge.zeros( (1,), dtype=diverge.rs_vertex_t )
# ( channel, [Rx,Ry,Rz], o1,o2, s1,s2,s3,s4, (V_real,V_imag) )
vertices[0] = ( 'D', [0,0,0], 0,0, -1,0,0,0, (3.0,0.0) )

diverge.autoflow( hoppings, vertices, nk=[20,20,0] )
