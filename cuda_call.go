package cudago

// CUDARetType for CUDA function return value type
//
// Now support cudaError_t, char * and const char *
type CUDARetType int

const (
	RetCUDAErrt = CUDARetType(0) // return type: cudaError_t
	RetString   = CUDARetType(1) // return type: char */const char *
)

// CudaCall interface for call CUDA library function
type CudaCall interface {
	// CallCUDAFuncRetInt Call CUDA library function with return type cudaError_t
	CallCUDAFuncRetInt(funcName string, p ...interface{}) (r int, err error)
	// CallCUDAFuncRetString Call CUDA library function with return type char */const char *
	CallCUDAFuncRetString(funcName string, p ...interface{}) (r string, err error)
}
