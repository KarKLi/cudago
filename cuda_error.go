package cudago

import (
	"fmt"
)

// CudaError implments error interface and records CUDA error info.
type CudaError struct {
	ErrCode int
	ErrName string
	ErrMsg  string
}

func (e *CudaError) Error() string {
	return fmt.Sprintf("errcode: %d(%s), errmsg: %s", e.ErrCode, e.ErrName, e.ErrMsg)
}
