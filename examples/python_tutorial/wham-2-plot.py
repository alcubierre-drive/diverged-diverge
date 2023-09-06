#!/usr/bin/env python3
#coding:utf8

import numpy as np
import matplotlib.pyplot as plt
import diverge.output as ro

model_out = ro.read("nickelate_model.dvg")
plt.plot( model_out.bandstructure[:,:-3], c='k' )
plt.show()
