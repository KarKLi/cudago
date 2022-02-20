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
