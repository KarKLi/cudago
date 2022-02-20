package windows

import (
	"github.com/karkli/cudago"
)

func (l *CudaRTLib) CudaGetErrorName(errCode CUDAError_t) (string, error) {
	return callCUDAFuncRetString(l.d, "cudaGetErrorName", uintptr(errCode))
}

func (l *CudaRTLib) CudaGetErrorString(errCode CUDAError_t) (string, error) {
	return callCUDAFuncRetString(l.d, "cudaGetErrorString", uintptr(errCode))
}

func cudaErrorHandler(l *CudaRTLib, errCode CUDAError_t) error {
	errName, err := l.CudaGetErrorName(errCode)
	if err != nil {
		panic(err)
	}
	msg, err := l.CudaGetErrorString(errCode)
	if err != nil {
		panic(err)
	}
	return &cudago.CudaError{
		ErrCode: int(errCode),
		ErrName: errName,
		ErrMsg:  msg,
	}
}
