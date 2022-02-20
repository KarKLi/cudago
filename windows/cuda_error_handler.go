package windows

import (
	"github.com/karkli/cudago"
)

// CudaGetErrorName - CUDA library host function
//
// const char* cudaGetErrorName(cudaError_t error);
func (l *CudaRTLib) CudaGetErrorName(errCode CUDAError_t) (string, error) {
	caller := NewCUDAWindowsCall(l.d)
	return caller.CallCUDAFuncRetString("cudaGetErrorName", errCode)
}

// CudaGetErrorString - CUDA library host function
//
// const char* cudaGetErrorString(cudaError_t error);
func (l *CudaRTLib) CudaGetErrorString(errCode CUDAError_t) (string, error) {
	caller := NewCUDAWindowsCall(l.d)
	return caller.CallCUDAFuncRetString("cudaGetErrorString", errCode)
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
