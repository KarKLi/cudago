package windows

import (
	"github.com/KarKLi/cudago"
)

// CudaGetErrorName - CUDA library host function
//
// const char* cudaGetErrorName(cudaError_t error);
func (c *cudaWindowsCall) CudaGetErrorName(errCode CUDAError_t) (string, error) {
	return c.CallCUDAFuncRetString("cudaGetErrorName", errCode)
}

// CudaGetErrorString - CUDA library host function
//
// const char* cudaGetErrorString(cudaError_t error);
func (c *cudaWindowsCall) CudaGetErrorString(errCode CUDAError_t) (string, error) {
	return c.CallCUDAFuncRetString("cudaGetErrorString", errCode)
}

func cudaErrorHandler(c *cudaWindowsCall, errCode CUDAError_t) error {
	errName, err := c.CudaGetErrorName(errCode)
	if err != nil {
		panic(err)
	}
	msg, err := c.CudaGetErrorString(errCode)
	if err != nil {
		panic(err)
	}
	return &cudago.CudaError{
		ErrCode: int(errCode),
		ErrName: errName,
		ErrMsg:  msg,
	}
}
