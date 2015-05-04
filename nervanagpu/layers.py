# Copyright 2014 Nervana Systems Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pycuda.driver as drv
from pycuda.tools import context_dependent_memoize
from operator import mul
from math     import ceil

class Layer(object):
    def __init__(self, lib, dtype, N):
        self.N         = N
        self.dtype     = dtype
        self.lib       = lib
        self.fprop_in  = None
        self.fprop_out = None
        self.bprop_in  = None
        self.bprop_out = None
        self.weights   = None
        self.updates   = None
        self.velocity  = None
        self.dimF2 = None
        self.flops = 0
        self.sizeO = 0
        self.sizeF = 0

    def init_activations(self):
        self.fprop_out    = self.lib.empty(self.dimO2, dtype=self.dtype)
        self.fprop_out_ew = self.fprop_out.reshape(self.dimOew)

        self.act_stats1 = self.lib.empty((self.dimOew[0],1), dtype=np.float32)
        self.act_stats2 = self.act_stats1[0:1,0:1]

    def init_deltas(self, shared=None):
        if shared is None:
            self.bprop_in = self.lib.empty(self.dimO2, dtype=self.dtype)
        else:
            self.bprop_in = shared.share(self.dimO2)

        self.bprop_in_ew = self.bprop_in.reshape(self.dimOew)

    def init_weights(self, loc=0.0, scale=0.1, shared=None):
        if self.dimF2 is not None:
            weights = np.random.normal(loc, scale, self.dimF2)
            self.weights  = self.lib.array(weights,    dtype=self.dtype)
            self.velocity = self.lib.zeros(self.dimF2, dtype=self.dtype)
            if shared is None: 
                self.updates = self.lib.empty(self.dimF2, dtype=self.dtype)
            else:
                self.updates = shared.share(self.dimF2)

            self.weights_ew  = self.weights.reshape(self.dimFew)
            self.updates_ew  = self.updates.reshape(self.dimFew)
            self.velocity_ew = self.velocity.reshape(self.dimFew)

            self.weight_stats1 = self.lib.empty((self.dimFew[0],1), dtype=np.float32)
            self.weight_stats2 = self.weight_stats1[0:1,0:1]

    def connect(self, prev_layer):
        if prev_layer is not None:
            self.fprop_in  = prev_layer.fprop_out
            self.bprop_out = prev_layer.bprop_in

    def reduction_factor(self):
        return 1.0

    def fprop(self): pass
    def bprop(self): pass
    def update(self, momentum, learning_rate): pass

    # fprop relu happens inside of the conv and gemm kernels
    def bprop_relu(self):

        self.bprop_in_ew *= self.fprop_out_ew > 0

    def grad_descent_momentum(self, momentum, learning_rate):

        self.velocity_ew[:] = self.velocity_ew*momentum - self.updates_ew*learning_rate
        self.weights_ew += self.velocity_ew

    def get_activation_mean(self):
        return self._get_mean(self.fprop_out_ew, self.act_stats1, self.act_stats2)

    def get_delta_mean(self, mean=False):
        return self._get_mean(self.bprop_in_ew, self.act_stats1, self.act_stats2)

    def get_update_mean(self, mean=False):
        if self.dimF2 is not None:
            return self._get_mean(self.updates_ew, self.weight_stats1, self.weight_stats2)
        return self._get_mean(self.bprop_in_ew, self.act_stats1, self.act_stats2)

    def get_weight_mean(self, mean=False):
        if self.dimF2 is not None:
            return self._get_mean(self.weights_ew, self.weight_stats1, self.weight_stats2)

    def get_activation_max(self):
        return self._get_max(self.fprop_out_ew, self.act_stats1, self.act_stats2)

    def get_delta_max(self, mean=False):
        return self._get_max(self.bprop_in_ew, self.act_stats1, self.act_stats2)

    def get_update_max(self, mean=False):
        if self.dimF2 is not None:
            return self._get_max(self.updates_ew, self.weight_stats1, self.weight_stats2)

    def get_weight_max(self, mean=False):
        if self.dimF2 is not None:
            return self._get_max(self.weights_ew, self.weight_stats1, self.weight_stats2)

    def _get_mean(self, ary, buf1, buf2):
        return float(self.lib.mean(abs(ary), partial=buf1, out=buf2).get()[0,0])

    def _get_max(self, ary, buf1, buf2):
        return float(self.lib.max(abs(ary), partial=buf1, out=buf2).get()[0,0])

class DataLayer(Layer):
    def __init__(self, lib, dtype, N, C, D=1, H=1, W=1):

        super(DataLayer, self).__init__(lib, dtype, N)

        self.C = C
        self.K = C
        self.M = D
        self.P = H
        self.Q = W
        self.DHW = (D,H,W)
        self.dimO2  = (C*D*H*W,N)
        self.dimOew = (C*D*H,W*N)

    def init_data(self, ary):
        self.fprop_out.set(ary)

    def init_deltas(self, shared=None):  pass
    def init_weights(self, loc=0.0, scale=0.1, shared=None): pass

    def __str__(self):
        return "DataLayer: NCK: (%d, %d, %d) DHW:%s" % (self.N, self.C, self.K, self.DHW)

class FullLayer(Layer):
    def __init__(self, lib, dtype, N, nIn, nOut, fprop_size=None, bprop_size=None):

        super(FullLayer, self).__init__(lib, dtype, N)

        self.nIn    = nIn
        self.nOut   = nOut
        self.flops  = N * nIn * nOut * 2.0
        self.dimF2  = (nOut, nIn)
        self.dimFew = (nOut, nIn)
        self.dimO2  = (nOut, N)
        self.sizeO  = nOut * N
        self.sizeF  = nIn * nOut
        div = 1
        min_blocks = _get_sm_count() * 8
        for d in range(64,1,-1):
            if nOut % d == 0 and nOut / d > min_blocks:
                div = d
                break
        self.dimOew = (nOut/div, N*div)

        self.fprop_size = fprop_size
        self.bprop_size = bprop_size

    def fprop(self):

        self.lib.dot(self.weights, self.fprop_in, self.fprop_out, relu=True, size=self.fprop_size)

    def bprop(self):

        self.bprop_relu()
        self.lib.dot(self.weights.T, self.bprop_in, self.bprop_out, size=self.bprop_size)

    def update(self, momentum, learning_rate):

        self.lib.dot(self.bprop_in, self.fprop_in.T, self.updates)
        self.grad_descent_momentum(momentum, learning_rate)

    def __str__(self):
        return "FullLayer: N, nIn, nOut: (%d, %d, %d)" % (self.N, self.nIn, self.nOut)

class ConvLayer(Layer):

    def __init__(self, lib, dtype,
            N, C, K,
            D=1, H=1, W=1,
            T=1, R=1, S=1,
            pad_d=0, pad_h=0, pad_w=0,
            str_d=1, str_h=1, str_w=1,
            grid_P=0, grid_Q=0, update_size=None):

        super(ConvLayer, self).__init__(lib, dtype, N)

        assert N % 8 == 0, "N dim must be multiple of 8"
        assert K % 8 == 0, "K dim must be multiple of 8"

        # Compute the output spatial dimensions
        M = int(ceil(float(D - T + 1 + 2*pad_d) / str_d))
        P = int(ceil(float(H - R + 1 + 2*pad_h) / str_h))
        Q = int(ceil(float(W - S + 1 + 2*pad_w) / str_w))

        self.C = C
        self.K = K
        self.M = M
        self.P = P
        self.Q = Q
        self.NCK = (N,C,K)
        self.TRS = (T,R,S)
        self.DHW = (D,H,W)
        self.MPQ = (M,P,Q)
        self.padding = (pad_d, pad_h, pad_w)
        self.strides = (str_d, str_h, str_w)

        self.dimI   = (C,D,H,W,N)
        self.dimF   = (C,T,R,S,K)
        self.dimO   = (K,M,P,Q,N)
        self.dimI2  = (C*D*H*W,N)
        self.dimF2  = (C*T*R*S,K)
        self.dimO2  = (K*M*P*Q,N)
        self.dimIew = (C*D*H,W*N)
        self.dimFew = (C*T*R,S*K)
        self.dimOew = (K*M*P,Q*N)
        self.sizeI  = reduce(mul, self.dimI, 1)
        self.sizeF  = reduce(mul, self.dimF, 1)
        self.sizeO  = reduce(mul, self.dimO, 1)
        self.nOut   = reduce(mul, self.MPQ,  1) * K

        # precompute some multiplications for fast constant memory access
        WN   = W*N
        HWN  = H*WN
        DHWN = D*HWN
        RS   = R*S
        RST  = T*RS
        CRST = C*RST
        PQ   = P*Q
        PM   = P*M
        PQM  = M*PQ
        QN   = Q*N
        PQN  = P*QN
        MPQN = M*PQN

        # I can easily get the kernels working with larger values here.. 
        # But this is what version 1 is coded to support.
        assert PQM < 2**16, "Integer division is faster with 16bit numerators"

        # Kernels can be recoded to support 32bit numerators at
        # some performance loss.
        assert CRST+8 < 2**16, "Integer division is faster with 16bit numerators"

        # precompute grid dimensions
        grid_N64  = N    // 64 + (N    % 64 != 0)
        grid_K64  = K    // 64 + (K    % 64 != 0)
        grid_C64  = CRST // 64 + (CRST % 64 != 0)

        grid_N128 = N    // 128 + (N    % 128 != 0)
        grid_K128 = K    // 128 + (K    % 128 != 0)
        grid_C128 = CRST // 128 + (CRST % 128 != 0)

        #TODO: add more 128x128 kernels for better performance at fp32.
        self.fprop_grid = (PQM, grid_K64,  grid_N64)
        self.bprop_grid = (PQM, grid_C128, grid_N64)
        self.fprop_block = (64,  1, 1)
        self.bprop_block = (128, 1, 1)
        self.fprop_size = "K64_N64"
        self.bprop_size = "C128_N64"

        #TODO: tune this further
        if  (update_size is None or update_size == "C64_K64" or update_size == "C128_K64") and \
            (CRST <= 64 or K <= 64 or (K % 64 == 0 and K % 128 != 0)):

            if self.dtype is np.float32:
                self.updat_size = "C128_K64"
                updat_grid  = [0, grid_C128, grid_K64]
                updat_block = 128
            else:
                self.updat_size = "C64_K64"
                updat_grid  = [0, grid_C64, grid_K64]
                updat_block = 64
        else:
            self.updat_size = "C128_K128"
            updat_grid  = [0, grid_C128, grid_K128]
            updat_block = 256

        if grid_P == 0 or grid_Q == 0:
            # Performance seems good with at least 4096 total threads per SM
            # More threads might be faster but accuracy starts dropping off.
            # Cap grid_P*grid_Q at 64 for fp16.
            # TODO: explore L2 utilization here:
            if self.dtype is np.float16:
                inc_P  = False
                grid_P = 1
                grid_Q = 1
                grid_O = updat_grid[1] * updat_grid[2] * M * updat_block
                thresh = _get_sm_count() * 4096
                while grid_O * grid_P * grid_Q < thresh and \
                      grid_P <= P and grid_Q <= Q and \
                      grid_P * grid_Q < 64:
                    if inc_P:
                        grid_P += 1
                    else:
                        grid_Q += 1
                    inc_P = not inc_P
            # When not concerned about accumulation accuracy just unroll things a bit
            # but maximize the distribution.  This has the effect of better utilizing the L2.
            else:
                grid_P = P
                grid_Q = Q // 4

            # TitanX optimization: make grid multiple of 24 for small grids
            # TODO: explore L2 utilization here:
            # TODO: add 980, 750, etc optimizations
            if _get_sm_count() == 24:
                grid_PQ  = grid_P * grid_Q
                if   grid_PQ < 30:
                    grid_P = 6
                    grid_Q = 4
                elif grid_PQ < 54:
                    grid_P = 8
                    grid_Q = 6
                elif grid_PQ < 78:
                    grid_P = 9
                    grid_Q = 8
                elif grid_PQ <= 108:
                    grid_P = 12
                    grid_Q = 8

        if grid_P >= P: grid_P = P
        if grid_Q >= Q: grid_Q = Q

        grid_PQ  = grid_P * grid_Q
        grid_PQM = updat_grid[0] = grid_PQ * M


        self.updat_grid  = tuple(updat_grid)
        self.updat_block = (updat_block,1,1)

        # precompute the magic numbers and shift amounts for integer division
        magic_RST = _magic32(CRST+8, RST)
        magic_RS  = _magic32(RST+32, RS)
        magic_S   = _magic32(RS+32,  S)
        magic_PQ  = _magic32(PQM, PQ)
        magic_Q   = _magic32(PQ,  Q)
        magic_PQu = _magic32(grid_PQM, grid_PQ)
        magic_Qu  = _magic32(grid_PQ,  grid_Q)

        # generate the convolution kernel args for fprop and bprop
        self.kernel_args = _flatten([
            N, K, D, H, W, WN, HWN, DHWN,
            C, CRST, RST, magic_RST, RS, magic_RS, S, magic_S,
            pad_d, pad_h, pad_w, str_d, str_h, str_w,
            P, Q, PQ, QN, PQN, MPQN, magic_Q, magic_PQ,
            grid_P, grid_Q, grid_PQ])

        # update uses slightly different args
        self.update_args = _flatten([
            N, K, D, H, W, WN, HWN, DHWN,
            C, CRST, RST, magic_RST, RS, magic_RS, S, magic_S,
            pad_d, pad_h, pad_w, str_d, str_h, str_w,
            P, Q, PQ, QN, PQN, MPQN, magic_Qu, magic_PQu,
            grid_P, grid_Q, grid_PQ])

        # shared lookup table size
        self.lut_size = (RST // 32 + (RST  % 32 != 0)) * 32 * 4

        # flop count for benchmarking
        self.flops    = PQM * K * N * CRST * 2.0

    def fprop(self):
            self.lib.fprop_conv(self, self.fprop_in, self.weights, self.fprop_out, relu=True)

    def bprop(self):
            self.bprop_relu()
            if self.bprop_out is not None:
                self.lib.bprop_conv(self, self.weights, self.bprop_in, self.bprop_out)

    def update(self, momentum, learning_rate):
            self.lib.update_conv(self, self.fprop_in, self.bprop_in, self.updates)
            self.grad_descent_momentum(momentum, learning_rate)

    def __str__(self):
        return "ConvLayer: NCK: (%d, %d, %d) DHW:%s TRS:%s MPQ:%s" % \
                (self.N, self.C, self.K, self.DHW, self.TRS, self.MPQ)

class PoolLayer(Layer):

    def __init__(self, lib, dtype,
            op, N, C,
            D=1, H=1, W=1,
            J=1, T=1, R=1, S=1,
            pad_j=0, pad_d=0, pad_h=0, pad_w=0,
            str_j=None, str_d=None, str_h=None, str_w=None):

        super(PoolLayer, self).__init__(lib, dtype, N)

        # default to non-overlapping
        if str_j is None: str_j = J
        if str_d is None: str_d = T
        if str_h is None: str_h = R
        if str_w is None: str_w = S

        # Compute the output dimensions
        K = int(ceil(float(C - J + 1 + 2*pad_j) / str_j))
        M = int(ceil(float(D - T + 1 + 2*pad_d) / str_d))
        P = int(ceil(float(H - R + 1 + 2*pad_h) / str_h))
        Q = int(ceil(float(W - S + 1 + 2*pad_w) / str_w))

        self.op   = op
        self.C    = C
        self.K    = K
        self.M    = M
        self.P    = P
        self.Q    = Q
        self.JTRS = (J,T,R,S)
        self.DHW  = (D,H,W)
        self.MPQ  = (M,P,Q)
        self.padding = (pad_j, pad_d, pad_h, pad_w)
        self.strides = (str_j, str_d, str_h, str_w)

        self.dimI   = (C,D,H,W,N)
        self.dimO   = (K,M,P,Q,N)
        self.dimF2  = None
        self.dimI2  = (C*D*H*W,N)
        self.dimO2  = (K*M*P*Q,N)
        self.dimIew = (C*D*H,W*N)
        self.dimOew = (K*M*P,Q*N)
        self.sizeI  = reduce(mul, self.dimI, 1)
        self.sizeO  = reduce(mul, self.dimO, 1)
        self.nOut   = reduce(mul, self.MPQ, 1) * K

        # precompute some multiplications for fast constant memory access
        WN   = W*N
        HWN  = H*WN
        DHWN = D*HWN
        RS   = R*S
        RST  = T*RS
        JRST = J*RST
        QN   = Q*N
        PM   = P*M
        PQN  = P*QN
        MPQN = M*PQN

        assert JRST <= N or N >= 32, "Edge case not currently implemented"
        assert JRST+32 < 2**16, "Integer division is faster with 16bit numerators"

        # precompute the magic numbers and shift amounts for integer division
        magic_RST   = _magic32(JRST+32, RST)
        magic_RS    = _magic32( RST+32, RS)
        magic_S     = _magic32(  RS+32, S)
        magic_P     = _magic32(PM,  P)

        # generate the convolution kernel args for all three operations
        self.kernel_args = _flatten([
            N, W, H, D, C, WN, HWN, DHWN,
            P, magic_P, QN, PQN, MPQN,
            pad_j, pad_d, pad_h, pad_w,
            str_j, str_d, str_h, str_w,
            S, RS, RST, JRST, magic_S, magic_RS, magic_RST])

        # precompute grid dimensions
        self.grid  = (Q, PM, K)
        self.block = (N, 1,  1)

        # shared lookup table size
        self.lut_size = (JRST // 32 + (JRST  % 32 != 0)) * 32 * 4

    def fprop(self):
        self.lib.fprop_pool(self, self.fprop_in, self.fprop_out)

    def bprop(self):
        self.lib.bprop_pool(self, self.fprop_in, self.bprop_in, self.bprop_out)

    def reduction_factor(self):
        return float(self.dimI2[0]) / float(self.dimO2[0])

    def __str__(self):
        return "PoolLayer: NCK: (%d, %d, %d) DHW:%s JTRS:%s MPQ:%s op: %s " % \
                (self.N, self.C, self.K, self.DHW, self.JTRS, self.MPQ, self.op)

# Magic numbers and shift amounts for integer division
# Suitable for when nmax*magic fits in 32 bits
# Shamelessly pulled directly from:
# http://www.hackersdelight.org/hdcodetxt/magicgu.py.txt
def _magic32(nmax, d):
    nc = ((nmax + 1)//d)*d - 1
    nbits = len(bin(nmax)) - 2
    for p in range(0, 2*nbits + 1):
        if 2**p > nc*(d - 1 - (2**p - 1)%d):
            m = (2**p + d - 1 - (2**p - 1)%d)//d
            return (m, p)
    raise ValueError("Can't find magic number for division")

# flatten a nested list of lists or values
def _flatten(lst):
    return sum( ([x] if not isinstance(x, (list,tuple))
                 else _flatten(x) for x in lst), [] )

@context_dependent_memoize
def _get_sm_count():
    attributes = drv.Context.get_device().get_attributes()
    return attributes[drv.device_attribute.MULTIPROCESSOR_COUNT]
