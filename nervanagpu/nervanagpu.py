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

import os
import numpy as np
import pycuda.driver as drv
from pycuda.tools import context_dependent_memoize
from struct import unpack_from
from pytools import memoize, memoize_method
from float_ew import call_compound_kernel
from layers import DataLayer, FullLayer, ConvLayer, PoolLayer, _get_sm_count


class GPUTensor(object):

    def __init__(self, shape,
                dtype     = np.float16,
                allocator = drv.mem_alloc,
                base      = None,
                gpudata   = None,
                strides   = None,
                is_trans  = False,
                name      = None,
                rounding  = 0):

        # supported dtypes
        assert dtype in (np.float16, np.float32, np.uint8, np.int8)

        dtype = np.dtype(dtype)

        try:
            size = 1
            for dim in shape:
                size *= dim
        except TypeError:
            assert isinstance(shape, (int, long, np.integer))
            size  = shape
            shape = (shape,)

        if isinstance(size, np.integer):
            size = np.asscalar(size)

        # only support C ordering for now.
        if strides is None:
            self.strides = _contiguous_strides(dtype.itemsize, shape)
        else:
            self.strides = tuple(strides)

        self.base       = base
        self.shape      = shape
        self.size       = size
        self.dtype      = dtype
        self.nbytes     = dtype.itemsize * size
        self.allocator  = allocator
        self.is_trans   = is_trans
        self.name       = name
        self.rounding   = rounding

        if gpudata is None:
            if size:
                #print drv.mem_get_info()
                self.gpudata = allocator(self.nbytes)
            else:
                self.gpudata = None

            assert base is None
        else:
            self.gpudata = gpudata

    def __str__(self):
        return ("Array(0x%x) name:%s dtype:%s shape:%s strides:%s "
                " is_trans:%s" % (self.gpudata, self.name, self.dtype,
                self.shape, self.strides, self.is_trans))

    def __getstate__(self):
        """
        Defines what and how we go about serializing an instance of this class.

        Returns:
            numpy.ndarray: Representation of the underlying
                           `cudanet.CUDAMatrix` tensor
        """
        statedict = {'numpydata': self.asnumpyarray(),
                     'shape':     self.shape,
                     'dtype':     self.dtype,
                     'strides':   self.strides,
                     'is_trans':  self.is_trans,
                     'name':      self.name}
        return statedict

    def __setstate__(self, statedict):
        """
        Defines how we go about deserializing into an instance of this class.

        Arguments:
            state (numpy.ndarray): Serialized representation of the underlying
                                   `cudanet.CUDAMatrix` tensor to be unpacked.
        """
        kwargs = {x: statedict[x] for x in statedict.keys()
                                  if x not in ('shape', 'numpydata')}
        import pycuda.autoinit  # TODO: Only create if it does not exist

        self.__init__(statedict['shape'], dtype=np.float32)
        self.fill(statedict['numpydata'])

    def __repr__(self):
        return self.__str__()

    def __int__(self):
        return int(self.gpudata)

    def __len__(self):
        """Return the size of the leading dimension of self."""
        if len(self.shape):
            return self.shape[0]
        else:
            return 0

    @property
    @memoize_method
    def is_contiguous(self):
        return self.strides == _contiguous_strides(self.dtype.itemsize, self.shape)

    def set(self, ary, device=None):
        """
        copy host array to device.
        Arguments:
            ary: host array, needs to be contiguous
            device: device id, if not the one attached to current context
        Returns:
            self
        """
        assert ary.size == self.size
        assert self.is_contiguous, "Array in set() must be contiguous"
        if ary.dtype is not self.dtype:
            ary = ary.astype(self.dtype)
        assert ary.strides == self.strides

        if device is None:
            drv.memcpy_htod(self.gpudata, ary)
        else:
            # with multithreaded datasets, make a context before copying
            # and destroy it again once done.
            ctx = drv.Device(device).make_context()
            drv.memcpy_htod(self.gpudata, ary)
            ctx.pop()
            del ctx

        return self

    def get(self):
        """
        copy device array to host.
        Returns:
            the host numpy array
        """
        assert self.is_contiguous, "Array in get() must be contiguous"
        ary = np.empty(self.shape, self.dtype)
        drv.memcpy_dtoh(ary, self.gpudata)
        return ary

    def asnumpyarray(self):
        """
        asnumpyarray is an alias of get(), needed for MOP compatibility
        """
        return self.get()

    def asbuffer(self):
        """
        asbuffer returns buffer interface to gpu data
        """
        return self.gpudata.as_buffer(self.nbytes)

    def __getitem__(self, index):
        """
        return a sliced view of an array
        """
        if not isinstance(index, tuple):
            index = (index,)

        new_shape = []
        new_offset = 0
        new_strides = []

        seen_ellipsis = False

        index_axis = 0
        array_axis = 0
        while index_axis < len(index):
            index_entry = index[index_axis]

            if array_axis > len(self.shape):
                raise IndexError("too many axes in index")

            if isinstance(index_entry, slice):
                start, stop, idx_stride = index_entry.indices(
                        self.shape[array_axis])

                array_stride = self.strides[array_axis]

                new_shape.append((stop-start)//idx_stride)
                new_strides.append(idx_stride*array_stride)
                new_offset += array_stride*start

                index_axis += 1
                array_axis += 1

            elif isinstance(index_entry, (int, np.integer)):
                array_shape = self.shape[array_axis]
                if index_entry < 0:
                    index_entry += array_shape

                if not (0 <= index_entry < array_shape):
                    raise IndexError(
                            "subindex in axis %d out of range" % index_axis)

                new_offset += self.strides[array_axis]*index_entry

                index_axis += 1
                array_axis += 1

            elif index_entry is Ellipsis:
                index_axis += 1

                remaining_index_count = len(index) - index_axis
                new_array_axis = len(self.shape) - remaining_index_count
                if new_array_axis < array_axis:
                    raise IndexError("invalid use of ellipsis in index")
                while array_axis < new_array_axis:
                    new_shape.append(self.shape[array_axis])
                    new_strides.append(self.strides[array_axis])
                    array_axis += 1

                if seen_ellipsis:
                    raise IndexError(
                            "more than one ellipsis not allowed in index")
                seen_ellipsis = True

            else:
                raise IndexError("invalid subindex in axis %d" % index_axis)

        while array_axis < len(self.shape):
            new_shape.append(self.shape[array_axis])
            new_strides.append(self.strides[array_axis])

            array_axis += 1

        return self.__class__(
                shape      = tuple(new_shape),
                dtype      = self.dtype,
                allocator  = self.allocator,
                base       = self,
                gpudata    = int(self.gpudata)+new_offset,
                strides    = tuple(new_strides),
                name       = self.name,
                rounding   = self.rounding)

    def _assign(self, value):

        if isinstance(value, (int, float)):

            # if we have a contiguous array, then use the speedy driver kernel
            if self.is_contiguous:

                value = self.dtype.type(value)

                if self.dtype.itemsize == 1:
                    drv.memset_d8( self.gpudata,
                                   unpack_from('B', value)[0],
                                   self.size)
                elif self.dtype.itemsize == 2:
                    drv.memset_d16(self.gpudata,
                                   unpack_from('H', value)[0],
                                   self.size)
                else:
                    drv.memset_d32(self.gpudata,
                                   unpack_from('I', value)[0],
                                   self.size)

            # otherwise use our copy kerel
            else:
                OpTreeNode.build("assign", self, value)

        elif isinstance(value, GPUTensor):
            # TODO: add an is_binary_compat like function
            if self.is_contiguous and value.is_contiguous and self.dtype == value.dtype:
                drv.memcpy_dtod(self.gpudata, value.gpudata, self.nbytes)
            else:
                OpTreeNode.build("assign", self, value)

        # collapse and execute an op tree as a kernel
        elif isinstance(value, OpTreeNode):
            OpTreeNode.build("assign", self, value)

        # assign to numpy array (same as set())
        elif isinstance(value, np.ndarray):
            self.set(value)

        else:
            raise TypeError("Invalid type for assignment: %s" % type(value))

        return self

    def __setitem__(self, index, value):

        self.__getitem__(index)._assign(value)

    def fill(self, value):
        return self._assign(value)


    def copy(self, a):
        return self._assign(a)

    def copy_from(self, a):
        """ alias of copy"""
        return self.set(a)

    def reshape(self, *shape):
        """
        return a reshaped view
        """
        if isinstance(shape[0], tuple) or isinstance(shape[0], list):
            shape = tuple(shape[0])

        if shape == self.shape:
            return self

        size = reduce(lambda x, y: x * y, shape, 1)
        if size != self.size:
            raise ValueError("total size of new array must be unchanged")

        if not self.is_contiguous:
            raise TypeError("reshaping of non-contigous "
                            "arrays is not yet supported")

        return self.__class__(
                shape      = shape,
                dtype      = self.dtype,
                allocator  = self.allocator,
                base       = self,
                gpudata    = self.gpudata,
                strides    = _contiguous_strides(self.dtype.itemsize, shape),
                name       = self.name,
                rounding   = self.rounding)


    def share(self, shape, dtype=None, name=None):
        """
        return a view: ary, where ary.size <= self.size
        Allows easy sharing of tempoary memory
        """
        size = reduce(lambda x, y: x * y, shape, 1)
        if size > self.size:
            raise ValueError("total size of new array must <= size of parent")

        if not self.is_contiguous:
            raise TypeError("sharing of non-contigous "
                            "arrays is not yet supported")

        if dtype is None:
            dtype = self.dtype
        else:
            dtype = np.dtype(dtype)

        return self.__class__(
                shape      = shape,
                dtype      = dtype,
                allocator  = self.allocator,
                base       = self,
                gpudata    = self.gpudata,
                strides    = _contiguous_strides(dtype.itemsize, shape),
                name       = name,
                rounding   = self.rounding)

    @property
    def T(self):
        """
        return a transposed view
        """
        return self.__class__(
                shape      = self.shape[::-1],
                dtype      = self.dtype,
                allocator  = self.allocator,
                base       = self,
                gpudata    = self.gpudata,
                strides    = self.strides[::-1],
                is_trans   = not self.is_trans,
                name       = self.name,
                rounding   = self.rounding)

    def __add__      (self, other): return OpTreeNode.build("add", self, other)
    def __sub__      (self, other): return OpTreeNode.build("sub", self, other)
    def __mul__      (self, other): return OpTreeNode.build("mul", self, other)
    def __div__      (self, other): return OpTreeNode.build("div", self, other)
    def __truediv__  (self, other): return OpTreeNode.build("div", self, other)
    def __pow__      (self, other): return OpTreeNode.build("pow", self, other)
    def __radd__     (self, other): return OpTreeNode.build("add", other, self)
    def __rsub__     (self, other): return OpTreeNode.build("sub", other, self)
    def __rmul__     (self, other): return OpTreeNode.build("mul", other, self)
    def __rdiv__     (self, other): return OpTreeNode.build("div", other, self)
    def __rtruediv__ (self, other): return OpTreeNode.build("div", other, self)
    def __rpow__     (self, other): return OpTreeNode.build("pow", other, self)
    def __eq__       (self, other): return OpTreeNode.build("eq",  self, other)
    def __ne__       (self, other): return OpTreeNode.build("ne",  self, other)
    def __lt__       (self, other): return OpTreeNode.build("lt",  self, other)
    def __le__       (self, other): return OpTreeNode.build("le",  self, other)
    def __gt__       (self, other): return OpTreeNode.build("gt",  self, other)
    def __ge__       (self, other): return OpTreeNode.build("ge",  self, other)
    def __abs__      (self):        return OpTreeNode.build("abs", self,  None)
    def __neg__      (self):        return OpTreeNode.build("neg", self,  None)

    def __iadd__     (self, other): return OpTreeNode.build("add", self, other, out=self)
    def __isub__     (self, other): return OpTreeNode.build("sub", self, other, out=self)
    def __imul__     (self, other): return OpTreeNode.build("mul", self, other, out=self)
    def __idiv__     (self, other): return OpTreeNode.build("div", self, other, out=self)
    def __itruediv__ (self, other): return OpTreeNode.build("div", self, other, out=self)
    def __ipow__     (self, other): return OpTreeNode.build("pow", self, other, out=self)


class NervanaGPU(object):

    def __init__(self, stochastic_round=False, bench=False,
                 cubin_path=os.path.join("kernels", "cubin")):
        self.round_mode = 1 if stochastic_round else 0
        self.cubin_path = os.path.join(os.path.dirname(__file__), cubin_path)
        self.bench = bench

    def empty(self, shape, dtype=np.float16, name=None, allocator=drv.mem_alloc):
        """
        allocate the space for a GPUTensor
        """
        return GPUTensor(shape, dtype, allocator=allocator,
                          name=name, rounding=self.round_mode)

    def array(self, ary, dtype=np.float16, name=None, allocator=drv.mem_alloc):
        """
        converts a numpy array to a GPUTensor
        """
        return GPUTensor(ary.shape, dtype, allocator=allocator,
                          name=name, rounding=self.round_mode).set(ary)

    def zeros(self, shape, dtype=np.float16, name=None, allocator=drv.mem_alloc):
        """
        Returns an array of the given shape and dtype filled with 0's.
        """
        return GPUTensor(shape, dtype, allocator=allocator,
                          name=name, rounding=self.round_mode)._assign(0)

    def ones(self, shape, dtype=np.float16, name=None, allocator=drv.mem_alloc):
        """
        Returns an array of the given shape and dtype filled with 1's.
        """
        return GPUTensor(shape, dtype, allocator,
                          name=name, rounding=self.round_mode)._assign(1)

    def empty_like(self, other_ary, name=None):
        """
        Returns an array with the same params as another
        """
        return GPUTensor(other_ary.shape, other_ary.dtype, other_ary.allocator,
                          name=name, rounding=self.round_mode)

    def conv_layer(self, dtype,
            N, C, K,
            D=1, H=1, W=1,
            T=1, R=1, S=1,
            pad_d=0, pad_h=0, pad_w=0,
            str_d=1, str_h=1, str_w=1,
            grid_P=0, grid_Q=0, update_size=None):
        """
        Create a new ConvLayer parameter object.
        This then is passed as an argument to all the convolution operations.

        N: Number of images in mini-batch
        C: Number of input feature maps
        K: Number of output feature maps

        D: Depth  of input image
        H: Height of input image
        W: Width  of input image

        T: Depth  of filter kernel
        R: Height of filter kernel
        S: Width  of filter kernel

        padding: amount of zero-padding around the given edge
        strides: factor to step the filters by in a given direction

        grid_P, grid_Q: For the update operation define the size of the grid
        to distribute the work accross SMs.  The smaller the grid, the deeper the
        MM and hence more accumulation is done in fp32.  The bigger the grid,
        the more the work can be evenly spanned accross the SMs, at the cost of
        needing more fp16 accumuation operations and increased error.

        Set to 1,1 for full fp32 accuracy
        Set to P,Q for maximal distribution of work acrross SMs
        Set to 0,0 for automactially calculated optimal balance (recommened).

        Tweaking these params can have a large impact on performance as the
        L2 cache utilization is greatly effected by them.

        update_size: override kernel size selection for update.
            "C64_K64"   (fp16 only)
            "C128_K64"  (fp32 only)
            "C128_K128" (both)

        dtype: need to know dtype to setup proper kernels and params.

        Maximum utilization is achieved when N, K and C*R*S*T is
        a multiple of 64
        """
        return ConvLayer(self, dtype, N, C, K, D, H, W, T, R, S,
            pad_d, pad_h, pad_w, str_d, str_h, str_w, grid_P, grid_Q, update_size)

    def fprop_conv(self, layer, I, F, O, alpha=1.0, relu=False, repeat=1):

        assert layer.sizeI == I.size
        assert layer.sizeF == F.size
        assert layer.sizeO == O.size

        return self._execute_conv(
            layer, "fprop", layer.fprop_size,
            layer.fprop_grid, layer.fprop_block, layer.kernel_args, layer.lut_size,
            I, F, O, alpha, relu, False, repeat)

    def bprop_conv(self, layer, F, E, grad_I, alpha=1.0, repeat=1):

        assert layer.sizeF == F.size
        assert layer.sizeO == E.size
        assert layer.sizeI == grad_I.size

        return self._execute_conv(
            layer, "bprop", layer.bprop_size,
            layer.bprop_grid, layer.bprop_block, layer.kernel_args, layer.lut_size,
            F, E, grad_I, alpha, False, True, repeat)

    def update_conv(self, layer, I, E, grad_F, alpha=1.0, repeat=1):

        assert layer.sizeI == I.size
        assert layer.sizeO == E.size
        assert layer.sizeF == grad_F.size

        return self._execute_conv(
            layer, "updat", layer.updat_size,
            layer.updat_grid, layer.updat_block, layer.update_args, 0,
            I, E, grad_F, alpha, False, True, repeat)

    def _execute_conv(self, layer, op, size, grid, block, args, shared, A, B, C, alpha, relu, zero, repeat):

        assert B.dtype == C.dtype

        clss  = "hconv" if C.dtype.type is np.float16 else "sconv"
        if   A.dtype.type is np.uint8: op += '_u8'
        elif A.dtype.type is np.int8:  op += '_s8'

        flags = 0
        if C.rounding: flags |= 1
        if relu:       flags |= 2

        kernel = _get_conv_kernel(self.cubin_path, clss, op, size)
        params = [grid, block, _get_rand_state(),
                  C.gpudata, A.gpudata, B.gpudata,
                  alpha, flags ]
        params.extend(args)

        # Warmup
        if repeat > 1:
            for r in range(max(repeat // 10, 1)):
                kernel.prepared_call(*params, shared_size=shared)

        if self.bench or repeat > 1:
            start, end = _get_events()
            start.record()

        for r in range(repeat):
            if zero: C.fill(0.0)
            kernel.prepared_call(*params, shared_size=shared)

        if self.bench or repeat > 1:
            end.record()
            end.synchronize()
            msecs  = end.time_since(start) / repeat
            gflops = layer.flops / (msecs * 1000000.0)
            print "%7.3f msecs %8.3f gflops (%s: %s) size:%s grid:%s" \
                  "" % (msecs, gflops, op, layer, size, grid)

    def pool_layer(self, dtype,
            op, N, C,
            D=1, H=1, W=1,
            J=1, T=1, R=1, S=1,
            pad_j=0, pad_d=0, pad_h=0, pad_w=0,
            str_j=None, str_d=None, str_h=None, str_w=None):
        """
        Create a new PoolLayer parameter object.
        This then is passed as an argument to all pooling kernels.

        op: max, avg, l2 pooling
        N: Number of images in mini-batch

        C: Number of input feature maps
        D: Depth  of input image
        H: Height of input image
        W: Width  of input image

        J: Size of feature map pooling window (maxout n_pieces)
        T: Depth  of pooling window
        R: Height of pooling window
        S: Width  of pooling window

        padding: amount of zero-padding around the given image or feature map edge
        strides: factor to step the window by in a given direction (overlap allowed)

        Leave spatial dimensions at 1 to allow feature map pooling in the fc layers.
        """
        # default to non-overlapping
        if str_j is None: str_j = J
        if str_d is None: str_d = T
        if str_h is None: str_h = R
        if str_w is None: str_w = S

        return PoolLayer(self, dtype, op, N, C, D, H, W, J, T, R, S,
            pad_j, pad_d, pad_h, pad_w, str_j, str_d, str_h, str_w)

    def fprop_pool(self, layer, I, O, repeat=1):

        assert layer.sizeI == I.size
        assert layer.sizeO == O.size

        return self._execute_pool(layer, I, O, None, 0, repeat)

    def bprop_pool(self, layer, I, E, grad_I, repeat=1):

        assert layer.sizeI == I.size
        assert layer.sizeO == E.size
        assert layer.sizeI == grad_I.size
        assert I.dtype     == grad_I.dtype

        return self._execute_pool(layer, I, E, grad_I, 1, repeat)

    def _execute_pool(self, layer, I, O, B, mode, repeat):

        assert I.dtype == O.dtype

        clss = "hpool" if I.dtype.type is np.float16 else "spool"

        b_data = 0 if B is None else B.gpudata

        kernel = _get_pool_kernel(self.cubin_path, clss, layer.op)
        params = [layer.grid, layer.block, I.gpudata, O.gpudata, b_data, mode]
        params.extend(layer.kernel_args)

        # Warmup
        if repeat > 1:
            for r in range(max(repeat // 10, 1)):
                kernel.prepared_call(*params, shared_size=layer.lut_size)

        if self.bench or repeat > 1:
            start, end = _get_events()
            start.record()

        for r in range(repeat):
            if mode: B.fill(0)
            kernel.prepared_call(*params, shared_size=layer.lut_size)

        if self.bench or repeat > 1:
            end.record()
            end.synchronize()
            msecs  = end.time_since(start) / repeat
            print "%7.3f msecs (%s) grid:%s" % (msecs, layer, layer.grid)

    def dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False, repeat=1, size=None):
        """
        C = alpha * A * B   + beta * C
        C = alpha * A.T * B + beta * C
        C = alpha * A * B.T + beta * C

        relu: if true applied before output (and prior to beta addition)

        size: one of 32, 64, 128.  Sometimes the fastest tiling isn't chosen for you.
        """
        assert A.dtype == B.dtype == C.dtype
        itemsize = C.dtype.itemsize

        # one dimention must be contiguous
        assert min(A.strides) == itemsize
        assert min(B.strides) == itemsize
        assert min(C.strides) == itemsize

        lda = max(A.strides) // itemsize
        ldb = max(B.strides) // itemsize
        ldc = max(C.strides) // itemsize

        opA = 't' if A.is_trans else 'n'
        opB = 't' if B.is_trans else 'n'
        op  = opA + opB
        assert op != "tt"

        m = A.shape[0]
        n = B.shape[1]
        k = A.shape[1]

        assert m == C.shape[0]
        assert n == C.shape[1]
        assert k == B.shape[0]

        gridA = m // 128 + (m % 128 != 0)

        if op == "nt":
            size = 128

        # Some basic tile size selection.
        # Your best bet is to benchmark your code with all 3 sizes
        # and manually fine tune the selection for each layer.
        if size is None:
            if n < 384-16:
                n128 = n % 128
                if 0 < n128 < 112:
                    if 48 < n128 <= 64:
                        n64  = n // 64
                        n64 *= gridA // _get_sm_count()
                        # nn_64 is only faster than nn_32 when occupancy is
                        # more than 1 warp per scheduler.
                        if n64 > 1 or op == "tn":
                            size = 64
                        else:
                            size = 32
                    else:
                        size = 32
                else:
                    size = 128
            # There's a large regime where 64 is faster, but it's hard to characterize
            else:
                size = 128

        gridB   = n // size + (n % size != 0)
        threads = 256 if size == 128 else 128
        size    = "128x%d" % size

        # nt and nn are more efficient with k%16==0
        if C.dtype.type is np.float16:
            clss = "hgemm"
            if  op == "tn" and m % 8  == 0 and n % 8 == 0 or \
                op == "nn" and k % 16 == 0 and n % 8 == 0 or \
                op == "nt" and k % 16 == 0:
                op += "_vec"
        else:
            clss = "sgemm"
            if  op == "tn" and m % 4  == 0 and n % 4 == 0 or \
                op == "nn" and k % 8  == 0 and n % 4 == 0 or \
                op == "nt" and k % 16 == 0:
                op += "_vec"

        flags = 0
        if C.rounding: flags |= 1
        if relu:       flags |= 2

        kernel = _get_gemm_kernel(self.cubin_path, clss, op, size)
        params = [
            (gridA,gridB,1), (threads,1,1), _get_rand_state(),
            A.gpudata, B.gpudata, C.gpudata,
            lda, ldb, ldc, m, n, k,
            alpha, beta, flags ]

        # Warmup
        if repeat > 1:
            for r in range(max(repeat // 10, 1)):
                kernel.prepared_call(*params)

        if self.bench or repeat > 1:
            start, end = _get_events()
            start.record()

        for r in range(repeat):
            kernel.prepared_call(*params)

        if self.bench or repeat > 1:
            end.record()
            end.synchronize()
            msecs = end.time_since(start) / repeat
            gflops = (m * n * k * 2.0) / (msecs * 1000000.0)
            print "%7.3f msecs %4.0f gflops (%s_%s: %d,%d,%d) size:%s grid:(%d,%d)" % \
                (msecs,gflops,clss,op,m,n,k, size,gridA,gridB)
            if repeat > 1:
                return gflops

        return C

    def add         (self, a, b, out=None): return OpTreeNode.build("add", a, b, out=out)
    def subtract    (self, a, b, out=None): return OpTreeNode.build("sub", a, b, out=out)
    def multiply    (self, a, b, out=None): return OpTreeNode.build("mul", a, b, out=out)
    def divide      (self, a, b, out=None): return OpTreeNode.build("div", a, b, out=out)
    def true_divide (self, a, b, out=None): return OpTreeNode.build("div", a, b, out=out)
    def power       (self, a, b, out=None): return OpTreeNode.build("pow", a, b, out=out)
    def reciprocal  (self, a,    out=None): return OpTreeNode.build("div", 1, a, out=out)

    def negative    (self, a, out=None): return OpTreeNode.build("neg",  a, None, out=out)
    def absolute    (self, a, out=None): return OpTreeNode.build("abs",  a, None, out=out)
    def fabs        (self, a, out=None): return OpTreeNode.build("abs",  a, None, out=out)

    def sqrt        (self, a, out=None): return OpTreeNode.build("sqrt", a, None, out=out)
    def square      (self, a, out=None): return OpTreeNode.build("sqr",  a, None, out=out)
    def exp         (self, a, out=None): return OpTreeNode.build("exp",  a, None, out=out)
    def exp2        (self, a, out=None): return OpTreeNode.build("exp2", a, None, out=out)
    def log         (self, a, out=None): return OpTreeNode.build("log",  a, None, out=out)
    def log2        (self, a, out=None): return OpTreeNode.build("log2", a, None, out=out)
    def sig         (self, a, out=None): return OpTreeNode.build("sig",  a, None, out=out)
    def sig2        (self, a, out=None): return OpTreeNode.build("sig2", a, None, out=out)
    def tanh        (self, a, out=None): return OpTreeNode.build("tanh", a, None, out=out)
    def tanh2       (self, a, out=None): return OpTreeNode.build("tanh2",a, None, out=out)

    def finite      (self, a, out=None): return OpTreeNode.build("finite", a, None, out=out)

    def equal         (self, a, b, out=None): return OpTreeNode.build("eq", a, b, out=out)
    def not_equal     (self, a, b, out=None): return OpTreeNode.build("ne", a, b, out=out)
    def less          (self, a, b, out=None): return OpTreeNode.build("lt", a, b, out=out)
    def less_equal    (self, a, b, out=None): return OpTreeNode.build("le", a, b, out=out)
    def greater       (self, a, b, out=None): return OpTreeNode.build("gt", a, b, out=out)
    def greater_equal (self, a, b, out=None): return OpTreeNode.build("ge", a, b, out=out)

    def maximum(self, a, b, out=None): return OpTreeNode.build("maximum", a, b, out=out)
    def minimum(self, a, b, out=None): return OpTreeNode.build("minimum", a, b, out=out)

    def clip(self, a, a_min, a_max, out=None):
        return self.minimum(self.maximum(a, a_min), a_max, out=out)

    def sum(self, a, axis=None, partial=None, out=None, keepdims=True):
        if axis is None:
            assert partial is not None
            return self.sum(self.sum(a, axis=1, out=partial), axis=0, out=out)
        return OpTreeNode.build("sum", a, None, axis=axis, out=out)

    def max(self, a, axis=None, partial=None, out=None, keepdims=True):
        if axis is None:
            assert partial is not None
            return self.max(self.max(a, axis=1, out=partial), axis=0, out=out)
        return OpTreeNode.build("max", a, None, axis=axis, out=out)

    def min(self, a, axis=None, partial=None, out=None, keepdims=True):
        if axis is None:
            assert partial is not None
            return self.min(self.min(a, axis=1, out=partial), axis=0, out=out)
        return OpTreeNode.build("min", a, None, axis=axis, out=out)

    def argmax(self, a, axis=1, out=None, keepdims=True):
        return OpTreeNode.build("argmax", a, None, axis=axis, out=out)

    def argmin(self, a, axis=1, out=None, keepdims=True):
        return OpTreeNode.build("argmin", a, None, axis=axis, out=out)

    def mean(self, a, axis=None, partial=None, out=None, keepdims=True):
        shape = OpTreeNode.shape(a)
        if axis is None:
            assert partial is not None
            return self.multiply(
                        self.sum(self.sum(a, axis=1, out=partial), axis=0),
                        1.0/(shape[0]*shape[1]),
                        out=out)
        return self.multiply(self.sum(a, axis=axis), 1.0/shape[axis], out=out)

    def var(self, a, axis=None, partial=None, out=None, keepdims=True):
        if axis is None:
            assert partial is not None
            return self.mean(
                    self.square(a - self.mean(a, axis=axis, partial=partial, out=partial[0:1,0:1])),
                    axis=axis, partial=partial, out=out)

        return self.mean(self.square(a - self.mean(a, axis=axis)), axis=axis, out=out)

    def std(self, a, axis=None, partial=None, out=None, keepdims=True):
        return self.sqrt(self.var(a, axis=axis, partial=partial, out=out))

    def rand(self, out=None): return OpTreeNode.build("rand", None, None, out=out)

    def dropout(self, keep=0.5, out=None):
        return self.less_equal(self.rand(), keep, out=out)


# For constructing an op tree used in lazy evaluation
class OpTreeNode(tuple):

    def __new__(cls, *args):

        return tuple.__new__(cls, args)

    @staticmethod
    def build(op, a, b, out=None, **kwargs):

        for arg in (a,b):
            if not isinstance(arg, (int, float, GPUTensor, OpTreeNode, type(None))):
                return NotImplemented

        op_dict = { "op" : op }
        op_dict.update(kwargs)

        node = OpTreeNode(op_dict, a, b)

        # execute explicit assignment
        if op == "assign":
            return node.execute()

        # passing in an out value counts as assignment
        if out is not None:
            return OpTreeNode({ "op" : "assign" }, out, node).execute()

        # delay execution until assignment
        return node

    def execute(self):

        stack = self.traverse(list())

        return call_compound_kernel(_get_rand_state(), *stack)

    # post order walk op tree and produce postfix stack
    def traverse(self, stack):

        # Left
        if type(self[1]) is OpTreeNode:
            self[1].traverse(stack)
        elif self[1] is not None:
            stack.append(self[1])

        # Right
        if type(self[2]) is OpTreeNode:
            self[2].traverse(stack)
        elif self[2] is not None:
            stack.append(self[2])

        stack.append(self[0])

        return stack

    @staticmethod
    def shape(node):

        if type(node) is GPUTensor:
            return node.shape

        if type(node) is OpTreeNode:

            max_shape = [1,1]
            stack = node.traverse(list())
            for item in stack:
                if type(item) is GPUTensor:
                    for i in range(2):
                        max_shape[i] = max(max_shape[i], item.shape[i])
            return tuple(max_shape)

        #scalar
        return (1,1)

    def __add__      (self, other): return self.build("add", self, other)
    def __sub__      (self, other): return self.build("sub", self, other)
    def __mul__      (self, other): return self.build("mul", self, other)
    def __div__      (self, other): return self.build("div", self, other)
    def __truediv__  (self, other): return self.build("div", self, other)
    def __pow__      (self, other): return self.build("pow", self, other)
    def __radd__     (self, other): return self.build("add", other, self)
    def __rsub__     (self, other): return self.build("sub", other, self)
    def __rmul__     (self, other): return self.build("mul", other, self)
    def __rdiv__     (self, other): return self.build("div", other, self)
    def __rtruediv__ (self, other): return self.build("div", other, self)
    def __rpow__     (self, other): return self.build("pow", other, self)
    def __eq__       (self, other): return self.build("eq",  self, other)
    def __ne__       (self, other): return self.build("ne",  self, other)
    def __lt__       (self, other): return self.build("lt",  self, other)
    def __le__       (self, other): return self.build("le",  self, other)
    def __gt__       (self, other): return self.build("gt",  self, other)
    def __ge__       (self, other): return self.build("ge",  self, other)
    def __abs__      (self):        return self.build("abs", self,  None)
    def __neg__      (self):        return self.build("neg", self,  None)

def _contiguous_strides(itemsize, shape):
    if shape:
        strides = [itemsize]
        for s in shape[:0:-1]:
            strides.append(strides[-1]*s)
        return tuple(strides[::-1])
    else:
        return ()

@context_dependent_memoize
def _get_rand_state():
    # initialize our common pool of randomness (1/4 MB):
    # MAX_THREADS_PER_MULTIPROCESSOR * 32 SMs (32 to be somewhat future proof
    # and power of two). This size is currently hardcoded in the kernels,
    # to be parameterized ...
    rand_init  = np.random.random_integers(0,2**32-1,(2048*32,)).astype(np.uint32)
    rand_state = drv.mem_alloc(rand_init.nbytes)
    drv.memcpy_htod(rand_state, rand_init)
    return rand_state

@context_dependent_memoize
def _get_events():
    return (drv.Event(), drv.Event())

@context_dependent_memoize
def _get_module(path, clss, op, size=None):

    size = "" if size is None else "_" + size
    cubin = "{0}_{1}{2}.cubin".format(clss, op, size)
    return drv.module_from_file(os.path.join(path, cubin))

@context_dependent_memoize
def _get_gemm_kernel(path, clss, op, size):
    module = _get_module(path, clss, op, size)
    kernel = "{0}_{1}_{2}".format(clss, op, size)
    func   = module.get_function(kernel)
    func.prepare("PPPPIIIIIIffI")
    #print "Loaded: ", kernel
    return func

@context_dependent_memoize
def _get_conv_kernel(path, clss, op, size):
    module = _get_module(path, clss, op, size)
    kernel = "{0}_{1}_{2}".format(clss, op, size)
    func   = module.get_function(kernel)
    func.prepare("PPPPfIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
    #print "Loaded: ", kernel
    return func

@context_dependent_memoize
def _get_pool_kernel(path, clss, op):

    module = _get_module(path, clss, op)
    kernel = "{0}_{1}".format(clss, op)
    func   = module.get_function(kernel)
    func.prepare("PPPIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
    #print "Loaded: ", kernel
    return func

