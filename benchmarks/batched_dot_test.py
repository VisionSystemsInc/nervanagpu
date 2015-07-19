#!/usr/bin/python

import numpy         as np
import pycuda.driver as drv
from nervanagpu      import NervanaGPU
from pycuda.autoinit import context
from ipdb            import set_trace

np.set_printoptions(threshold=8192*4, linewidth=600, formatter={'int':lambda x: "%2d" % x,'float':lambda x: "%2.0f" % x})

ng = NervanaGPU(stochastic_round=0, bench=1)

dtype  = np.float16
repeat = 100
ones   = 0
cpu    = 0
size   = 64

X = 50
N = 64
C = 3072
K = 3072

dimI = (X,C,N)
dimO = (X,K,N)
dimW = (K,C)

if ones:
    cpuI = np.ones(dimI, dtype=np.float32)
    cpuE = np.ones(dimO, dtype=np.float32)
    cpuW = np.ones(dimW, dtype=np.float32)
else:
    cpuI = np.random.uniform(-1.0, 1.0, dimI).astype(dtype).astype(np.float32)
    cpuE = np.random.uniform(-1.0, 1.0, dimO).astype(dtype).astype(np.float32)
    cpuW = np.random.uniform(-1.0, 1.0, dimW).astype(dtype).astype(np.float32)

devI = ng.array(cpuI, dtype=dtype)
devE = ng.array(cpuE, dtype=dtype)
devW = ng.array(cpuW, dtype=dtype)

devO = ng.empty(dimO, dtype=dtype)
devB = ng.empty(dimI, dtype=dtype)
devU = ng.empty(dimW, dtype=dtype)

ng.batched_dot(devW,   devI,   devO, repeat=repeat, size=size) # fprop
ng.batched_dot(devW.T, devE,   devB, repeat=repeat, size=size) # bprop
ng.batched_dot(devE,   devI.T, devU, repeat=repeat, size=size) # update

if cpu:

    cpuO = np.empty(dimO, dtype=np.float32)
    cpuB = np.empty(dimI, dtype=np.float32)
    cpuU = np.zeros(dimW, dtype=np.float32)

    for i in range(X):
        cpuO[i] = np.dot(cpuW,    cpuI[i]  ) # fprop
        cpuB[i] = np.dot(cpuW.T,  cpuE[i]  ) # bprop
        cpuU   += np.dot(cpuE[i], cpuI[i].T) # update

    diffO = abs(devO.get() - cpuO)
    diffB = abs(devB.get() - cpuB)
    diffU = abs(devU.get() - cpuU)
    print diffO.max()
    print diffB.max()
    print diffU.max()

exit()

