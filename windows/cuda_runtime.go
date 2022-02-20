/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

// package windows implements Cuda C Runtime library to Go.
package windows

import (
	"fmt"
	"runtime"
	"strconv"
	"syscall"
	"unsafe"

	"github.com/karkli/cudago"
)

func init() {
	if strconv.IntSize != 64 {
		panic("cuda go version only works on 64-bit system.")
	}
	if runtime.GOOS != "windows" {
		panic("this library implements for Windows platform")
	}
}

// VoidPtr for C type void *
type VoidPtr uintptr

// VoidPtrPtr for C type void **
type VoidPtrPtr uintptr

// CudaRTLib wraps a runtime dynamic library of CUDA
type CudaRTLib struct {
	d *syscall.DLL
}

// NewCudaLib returns a CudaRTLib with specified cuda dynamic library
func NewCudaRTLib(d *syscall.DLL) *CudaRTLib {
	// 先确认当前传入的d是否加载的10.0版本的cuda dll，否则报错
	var runtimeVersion int
	r1, err := callCUDAFuncRetInt(d, "cudaRuntimeGetVersion", uintptr(unsafe.Pointer(&runtimeVersion)))
	if err != nil {
		panic(err)
	}
	if r1 != 0 {
		panic(cudaErrorHandler(&CudaRTLib{d}, CUDAError_t(r1)))
	}
	if runtimeVersion != cudago.Version() {
		panic(fmt.Sprintf("cuda version not match, wants: %d, library returns: %d", cudago.Version(), runtimeVersion))
	}
	return &CudaRTLib{
		d: d,
	}
}

// CudaFree - CUDA library host function
//
// cudaError_t cudaDeviceReset(void);
func (l *CudaRTLib) CudaDeviceReset() error {
	r1, err := callCUDAFuncRetInt(l.d, "cudaDeviceReset")
	if err != nil {
		return err
	}
	if r1 != 0 {
		return fmt.Errorf("cudaDeviceReset error: %v", cudaErrorHandler(l, CUDAError_t(r1)))
	}
	return nil
}

// CudaDeviceSynchronize - CUDA library host function
//
// cudaError_t cudaDeviceSynchronize(void);
func (l *CudaRTLib) CudaDeviceSynchronize() error {
	r1, err := callCUDAFuncRetInt(l.d, "cudaDeviceSynchronize")
	if err != nil {
		return err
	}
	if r1 != 0 {
		return fmt.Errorf("cudaDeviceSynchronize error: %v", cudaErrorHandler(l, CUDAError_t(r1)))
	}
	return nil
}

// CudaDeviceSetLimit - CUDA library host function
//
// cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value);
func (l *CudaRTLib) CudaDeviceSetLimit(limit cudago.CudaLimit, value uint64) error {
	r1, err := callCUDAFuncRetInt(l.d, "cudaDeviceSetLimit", uintptr(limit), uintptr(value))
	if err != nil {
		return err
	}
	if r1 != 0 {
		return fmt.Errorf("cudaDeviceSetLimit error: %v", cudaErrorHandler(l, CUDAError_t(r1)))
	}
	return nil
}

// CudaDeviceGetLimit - CUDA library host function
//
// cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit);
func (l *CudaRTLib) CudaDeviceGetLimit(pValue *uint64, limit cudago.CudaLimit) error {
	r1, err := callCUDAFuncRetInt(l.d, "cudaDeviceGetLimit", uintptr(unsafe.Pointer(pValue)), uintptr(limit))
	if err != nil {
		return err
	}
	if r1 != 0 {
		return fmt.Errorf("cudaDeviceSetLimit error: %v", cudaErrorHandler(l, CUDAError_t(r1)))
	}
	return nil
}

// CudaDeviceGetCacheConfig - CUDA library host function
//
// cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig);
func (l *CudaRTLib) CudaDeviceGetCacheConfig(pCacheConfig *cudago.CudaFuncCache) error {
	r1, err := callCUDAFuncRetInt(l.d, "cudaDeviceGetCacheConfig", uintptr(unsafe.Pointer(pCacheConfig)))
	if err != nil {
		return err
	}
	if r1 != 0 {
		return fmt.Errorf("cudaDeviceSetLimit error: %v", cudaErrorHandler(l, CUDAError_t(r1)))
	}
	return nil
}

// CudaMalloc - CUDA library host function
//
// cudaError_t cudaMalloc(void **devPtr, size_t size);
func (l *CudaRTLib) CudaMalloc(devPtr VoidPtrPtr, size uint64) error {
	r1, err := callCUDAFuncRetInt(l.d, "cudaMalloc", uintptr(devPtr), uintptr(size))
	if err != nil {
		return err
	}
	if r1 != 0 {
		return fmt.Errorf("cudaMalloc error: %v", cudaErrorHandler(l, CUDAError_t(r1)))
	}
	return nil
}

// CudaMemcpy - CUDA library host function
//
// cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
func (l *CudaRTLib) CudaMemcpy(dst VoidPtr, src VoidPtr, count uint64, kind cudago.CUDAMemcpyKind) error {
	r1, err := callCUDAFuncRetInt(l.d, "cudaMemcpy", uintptr(dst), uintptr(src), uintptr(count), uintptr(kind))
	if err != nil {
		return err
	}
	if r1 != 0 {
		return fmt.Errorf("cudaMemcpy error: %v", cudaErrorHandler(l, CUDAError_t(r1)))
	}
	return nil
}

// CudaFree - CUDA library host function
//
// cudaError_t cudaFree(void *devPtr);
func (l *CudaRTLib) CudaFree(devPtr VoidPtr) error {
	r1, err := callCUDAFuncRetInt(l.d, "cudaFree", uintptr(devPtr))
	if err != nil {
		return err
	}
	if r1 != 0 {
		return fmt.Errorf("cudaFree error: %v", cudaErrorHandler(l, CUDAError_t(r1)))
	}
	return nil
}
