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

// package cudago implements translation between CUDA C library and Go.
package cudago

// CUDAMemcpyKind - CUDA library enum type cudaMemcpyKind
type CUDAMemcpyKind int

const (
	CUDAMemcpyHostToHost     = CUDAMemcpyKind(0) /**< Host   -> Host */
	CUDAMemcpyHostToDevice   = CUDAMemcpyKind(1) /**< Host   -> Device */
	CUDAMemcpyDeviceToHost   = CUDAMemcpyKind(2) /**< Device -> Host */
	CUDAMemcpyDeviceToDevice = CUDAMemcpyKind(3) /**< Device -> Device */
	CUDAMemcpyDefault        = CUDAMemcpyKind(4) /**< Direction of the traμsfer is inferred from the pointer values. Requires unified virtual addressing */
)

// CudaLimit - CUDA library enum type cudaLimit
type CudaLimit int

const (
	CudaLimitStackSize                    = CudaLimit(0x00) /**< GPU thread stack size */
	CudaLimitPrintfFifoSize               = CudaLimit(0x01) /**< GPU printf FIFO size */
	CudaLimitMallocHeapSize               = CudaLimit(0x02) /**< GPU malloc heap size */
	CudaLimitDevRuntimeSyncDepth          = CudaLimit(0x03) /**< GPU device runtime synchronize depth */
	CudaLimitDevRuntimePendingLaunchCount = CudaLimit(0x04) /**< GPU device runtime pending launch count */
	CudaLimitMaxL2FetchGranularity        = CudaLimit(0x05) /**< A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint */
)

// CudaFuncCache - CUDA library enum type cudaFuncCache
type CudaFuncCache int

const (
	CudaFuncCachePreferNone   = CudaFuncCache(0) /**< Default function cache configuration, no preference */
	CudaFuncCachePreferShared = CudaFuncCache(1) /**< Prefer larger shared memory and smaller L1 cache  */
	CudaFuncCachePreferL1     = CudaFuncCache(2) /**< Prefer larger L1 cache and smaller shared memory */
	CudaFuncCachePreferEqual  = CudaFuncCache(3) /**< Prefer equal size L1 cache and shared memory */
)
