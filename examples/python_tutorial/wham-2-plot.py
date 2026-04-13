#!/usr/bin/env python3
#coding:utf8

import numpy as np
import matplotlib.pyplot as plt
import diverge.output as do

model_out = do.read("nickelate_model.dvg")
plt.plot( do.bandstructure_xvals(model_out), do.bandstructure_bands(model_out), c='k' )
ticks = do.bandstructure_ticks(model_out)
plt.xticks( ticks, "GMXGZARZ" )
plt.xlim( ticks[0], ticks[-1] )
plt.ylabel( r'$\epsilon_b(\boldsymbol{k})$' )
plt.show()
