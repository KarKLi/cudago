package windows

import (
	"fmt"
	"syscall"
	"unsafe"
)

// CUDAError_t CUDA error type
type CUDAError_t int

// callCUDAFuncRetInt Call CUDA function with return value of type int (equals to C type int or enum)
func callCUDAFuncRetInt(d *syscall.DLL, funcName string, p ...uintptr) (r int, err error) {
	proc := d.MustFindProc(funcName)
	r1, _, errno := proc.Call(p...)
	if errno != syscall.Errno(0) {
		return 0, fmt.Errorf("errno %d", errno)
	}
	// 将其解析成int返回
	return int(r1), nil
}

// callCUDAFuncRetString Call CUDA function with return value of type string (equals to C type char * or const char *)
func callCUDAFuncRetString(d *syscall.DLL, funcName string, p ...uintptr) (r string, err error) {
	proc := d.MustFindProc(funcName)
	r1, _, errno := proc.Call(p...)
	if errno != syscall.Errno(0) {
		return "", fmt.Errorf("errno %d", errno)
	}
	// 不能直接返回r1，因为离开该函数后，r1所指向的内容就会失效
	// 将其解析成string返回
	temp := make([]byte, 0, 100)
	for {
		p := *(*byte)(unsafe.Pointer(r1))
		if p == 0 {
			return string(temp), nil
		}
		temp = append(temp, p)
		r1 += unsafe.Sizeof(byte(0))
	}
}
