/*
 * Copyright 2015 Baidu USA, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <vector>
#include <string>
#include <map>
#include <cuda.h>
#include <iostream>
#include <sstream>
#include <mutex>
#include "nervana_c_api.h"

std::map<CUdevice, int> nervana_sm_counts_;
std::map<std::string, CUfunction> nervana_kernels_;
std::vector<CUmodule> nervana_modules_;

//for when we need to modify the above data structures
std::mutex nervana_load_kernels_mutex_;
std::mutex nervana_sm_count_mutex_;

extern "C" bool nervana_loadKernels(const char* const base_path_cstr) {
    std::lock_guard<std::mutex> lock(nervana_load_kernels_mutex_);

    //better would be a vector<string>, but there is a bug in nvcc that prevents this
    // (bug report filed)
    std::string names[40] = {
        "hgemm_nn_vec_128x128",
        "hgemm_nn_128x128",
        "hgemm_nt_vec_128x128",
        "hgemm_nt_128x128",
        "hgemm_tn_vec_128x128",
        "hgemm_tn_128x128",
        "hgemm_nn_vec_128x64",
        "hgemm_nn_128x64",
        "hgemm_tn_vec_128x64",
        "hgemm_tn_128x64",
        "hgemm_nn_vec_128x32",
        "hgemm_nn_128x32",
        "hgemm_tn_vec_128x32",
        "hgemm_tn_128x32",
        "hgemm_nn_32x128",
        "hgemm_nn_vec_32x128",
        "sgemm_nn_vec_128x128",
        "sgemm_nn_128x128",
        "sgemm_nt_vec_128x128",
        "sgemm_nt_128x128",
        "sgemm_tn_vec_128x128",
        "sgemm_tn_128x128",
        "sgemm_nn_vec_128x64",
        "sgemm_nn_128x64",
        "sgemm_tn_vec_128x64",
        "sgemm_tn_128x64",
        "sgemm_nn_vec_128x32",
        "sgemm_nn_128x32",
        "sgemm_tn_vec_128x32",
        "sgemm_tn_128x32",
        "sgemm_nn_32x128",
        "sgemm_nn_vec_32x128",
        "hsgemm_nn_128x128",
        "hsgemm_nn_32x128",
        "hsgemm_nn_vec_128x128",
        "hsgemm_nn_vec_32x128",
        "hsgemm_nt_128x128",
        "hsgemm_nt_32x128",
        "hsgemm_nt_vec_128x128",
        "hsgemm_nt_vec_32x128"
    };

    std::string base_path(base_path_cstr);

    for (auto kernel : names) {
        if (nervana_kernels_.count(kernel) > 0)
            continue;

        CUmodule module;

        std::string path = base_path + kernel + std::string(".cubin");
        CUresult res = cuModuleLoad(&module, path.c_str());

        if (res != CUDA_SUCCESS) {
            std::cerr << "Failed to load: " << kernel << " " << res << std::endl;
            return false;
        }

        nervana_modules_.push_back(module);

        CUfunction function;
        res = cuModuleGetFunction(&function, module, kernel.c_str());
        if (res != CUDA_SUCCESS) {
            std::cerr << "Failed to extract: " << kernel << " " << res << std::endl;
            return false;
        }

        nervana_kernels_.insert(std::make_pair(kernel, function));
    }

    return true;
}

extern "C" bool nervana_unloadKernels() {
    std::lock_guard<std::mutex> lock(nervana_load_kernels_mutex_);
    while(nervana_modules_.size() > 0) {
        auto module = nervana_modules_.back();
        CUresult res = cuModuleUnload(module);

        nervana_modules_.pop_back();

        if (res != CUDA_SUCCESS)
            return false;
    }

    nervana_kernels_.clear();

    return true;
}

extern "C" size_t nervana_randStateSizeBytes() {
    return 2048 * 32 * sizeof(int);
}

extern "C" bool nervana_sgemm(float *A, float *B, float *C,
                              bool a_t, bool b_t,
                              int m, int n, int k,
                              int lda, int ldb, int ldc,
                              float alpha, float beta,
                              unsigned int *rand_state,
                              bool stochastic_round, bool apply_relu,
                              CUstream stream
                             )
{
    int sm_count;
    {
        std::lock_guard<std::mutex> lock(nervana_sm_count_mutex_);

        CUdevice device;
        CUresult res = cuCtxGetDevice(&device);
        if (res != CUDA_SUCCESS) {
            return false;
        }
        auto count = nervana_sm_counts_.find(device);
        if (count != nervana_sm_counts_.end()) {
            sm_count = count->second;
        }
        else {
            int pi;
            res = cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
            if (res != CUDA_SUCCESS) {
                return false;
            }
            sm_count = pi;
            nervana_sm_counts_[device] = pi;
        }
    }

    int gridA, gridB, threads;

    std::string name = "sgemm_";

    std::string trans;
    trans += a_t ? 't' : 'n';
    trans += b_t ? 't' : 'n';

    name += trans;

    if ( (trans == "tn" && m % 4 == 0 && n % 4 == 0) ||
         (trans == "nn" && k % 8 == 0 && n % 4 == 0) ||
         (trans == "nt" && k % 16 == 0)) {
         name += "_vec";
    }

    int sizeA = 0;
    int sizeB = 0;
    if (m < 128 && trans == "nn") { //use 32x128 kernels
        sizeA = 32;
        sizeB = 128;
        gridA = m / sizeA + (m % sizeA != 0);
        threads = 128;
    }
    else {
        sizeA = 128;
        gridA = m / sizeA + (m % sizeA != 0);

        if (trans == "nt")
            sizeB = 128;

        if (sizeB == 0) {
            if (n < 384 - 16) {
                int n128 = n % 128;
                if (n128 > 0 && n128 < 112) {
                    if (n128 > 48 && n128 <= 64) {
                        int n64 = n / 64;
                        n64 *= gridA / sm_count;
                        if (n64 > 1 || trans == "tn") {
                            sizeB = 64;
                        }
                        else {
                            sizeB = 32;
                        }
                    }
                    else {
                        sizeB = 32;
                    }
                }
                else {
                    sizeB = 128;
                }
            }
            else {
                sizeB = 128;
            }
        }
        threads = sizeB == 128 ? 256 : 128;
    }

    gridB = n / sizeB + (n % sizeB != 0);
    std::stringstream ss;
    ss << "_" << sizeA << "x" << sizeB;
    name += ss.str();

    int flags = 0;
    flags |= (stochastic_round << 0);
    flags |= (apply_relu << 1);

    void *args[13] = {&rand_state, &A, &B, &C, &lda, &ldb, &ldc, &m, &n, &k, &alpha, &beta, &flags};

    CUresult res = cuLaunchKernel(nervana_kernels_[name],
                                  gridA, gridB, 1,
                                  threads, 1, 1,
                                  0,
                                  stream, args, NULL);

    if (res != CUDA_SUCCESS) {
        std::cerr << "Error launching kernel " << name << " " << res << std::endl;
        return false;
    }

    return true;
}

extern "C" bool nervana_hgemm(short *A, short *B, short *C,
                              bool a_t, bool b_t,
                              int m, int n, int k,
                              int lda, int ldb, int ldc,
                              float alpha, float beta,
                              unsigned int *rand_state,
                              bool stochastic_round, bool apply_relu,
                              CUstream stream
                             )
{
    int sm_count;
    {
        std::lock_guard<std::mutex> lock(nervana_sm_count_mutex_);

        CUdevice device;
        CUresult res = cuCtxGetDevice(&device);
        if (res != CUDA_SUCCESS) {
            return false;
        }
        auto count = nervana_sm_counts_.find(device);
        if (count != nervana_sm_counts_.end()) {
            sm_count = count->second;
        }
        else {
            int pi;
            res = cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
            if (res != CUDA_SUCCESS) {
                return false;
            }
            sm_count = pi;
            nervana_sm_counts_[device] = pi;
        }
    }

    int gridA, gridB, threads;

    std::string name = "hgemm_";

    std::string trans;
    trans += a_t ? 't' : 'n';
    trans += b_t ? 't' : 'n';

    name += trans;

    if ( (trans == "tn" && m % 8 == 0 && n % 8 == 0) ||
         (trans == "nn" && k % 16 == 0 && n % 8 == 0) ||
         (trans == "nt" && k % 16 == 0)) {
         name += "_vec";
    }

    int sizeA = 0;
    int sizeB = 0;
    if (m < 128 && trans == "nn") { //use 32x128 kernels
        sizeA = 32;
        sizeB = 128;
        gridA = m / sizeA + (m % sizeA != 0);
        threads = 128;
    }
    else {
        sizeA = 128;
        gridA = m / sizeA + (m % sizeA != 0);

        if (trans == "nt")
            sizeB = 128;

        if (sizeB == 0) {
            if (n < 384 - 16) {
                int n128 = n % 128;
                if (n128 > 0 && n128 < 112) {
                    if (n128 > 48 && n128 <= 64) {
                        int n64 = n / 64;
                        n64 *= gridA / sm_count;
                        if (n64 > 1 || trans == "tn") {
                            sizeB = 64;
                        }
                        else {
                            sizeB = 32;
                        }
                    }
                    else {
                        sizeB = 32;
                    }
                }
                else {
                    sizeB = 128;
                }
            }
            else {
                sizeB = 128;
            }
        }
        threads = sizeB == 128 ? 256 : 128;
    }

    gridB = n / sizeB + (n % sizeB != 0);
    std::stringstream ss;
    ss << "_" << sizeA << "x" << sizeB;
    name += ss.str();

    int flags = 0;
    flags |= (stochastic_round << 0);
    flags |= (apply_relu << 1);

    void *args[13] = {&rand_state, &A, &B, &C, &lda, &ldb, &ldc, &m, &n, &k, &alpha, &beta, &flags};

    CUresult res = cuLaunchKernel(nervana_kernels_[name],
                                  gridA, gridB, 1,
                                  threads, 1, 1,
                                  0,
                                  stream, args, NULL);

    if (res != CUDA_SUCCESS) {
        std::cerr << "Error launching kernel " << name << " " << res << std::endl;
        return false;
    }

    return true;
}

extern "C" bool nervana_hsgemm(short *A, float *B, short *C,
                               bool a_t, bool b_t,
                               int m, int n, int k,
                               int lda, int ldb, int ldc,
                               float alpha, float beta,
                               unsigned int *rand_state,
                               bool stochastic_round, bool apply_relu,
                               CUstream stream
                              )
{
    if (a_t) {
        std::cerr << "The TN variant is not implemented for hsgemm" << std::endl;
        return false;
    }

    int gridA, gridB, threads;

    std::string name = "hsgemm_";

    std::string trans;
    trans += a_t ? 't' : 'n';
    trans += b_t ? 't' : 'n';

    name += trans;

    if ( (trans == "nn" && k % 16 == 0 && n % 8 == 0) ||
         (trans == "nt" && k % 16 == 0)) {
         name += "_vec";
    }

    int sizeA = 0;
    int sizeB = 0;
    if (m < 128 && trans == "nn") { //use 32x128 kernels
        sizeA = 32;
        sizeB = 128;
        gridA = m / sizeA + (m % sizeA != 0);
        threads = 128;
    }
    else {
        sizeA = 128;
        sizeB = 128;
        gridA = m / sizeA + (m % sizeA != 0);

        threads = 256;
    }

    gridB = n / sizeB + (n % sizeB != 0);
    std::stringstream ss;
    ss << "_" << sizeA << "x" << sizeB;
    name += ss.str();

    int flags = 0;
    flags |= (stochastic_round << 0);
    flags |= (apply_relu << 1);

    void *args[13] = {&rand_state, &A, &B, &C, &lda, &ldb, &ldc, &m, &n, &k, &alpha, &beta, &flags};

    CUresult res = cuLaunchKernel(nervana_kernels_[name],
                                  gridA, gridB, 1,
                                  threads, 1, 1,
                                  0,
                                  stream, args, NULL);

    if (res != CUDA_SUCCESS) {
        std::cerr << "Error launching kernel " << name << " " << res << std::endl;
        return false;
    }

    return true;
}
