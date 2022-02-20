package main

import (
	"fmt"

	"github.com/KarKLi/cudago"
	"github.com/KarKLi/cudago/windows"
)

func main() {
	c := windows.NewCUDAWindowsCall("")
	var a windows.VoidPtr
	r1, err := c.CallCUDAFuncRetInt("cudaMalloc", &a, 0xffff)
	if err != nil {
		panic(err)
	}
	if r1 != 0 {
		panic(windows.CUDAErrorHandler(c, windows.CUDAError_t(r1)))
	}
	r1, err = c.CallCUDAFuncRetInt("cudaMemcpy", a, "hello,world", len("hello,world"), cudago.CUDAMemcpyHostToDevice)
	if err != nil {
		panic(err)
	}
	if r1 != 0 {
		panic(windows.CUDAErrorHandler(c, windows.CUDAError_t(r1)))
	}
	t := make([]byte, 20)
	r1, err = c.CallCUDAFuncRetInt("cudaMemcpy", t, a, len("hello,world"), cudago.CUDAMemcpyDeviceToHost)
	if err != nil {
		panic(err)
	}
	if r1 != 0 {
		panic(windows.CUDAErrorHandler(c, windows.CUDAError_t(r1)))
	}
	fmt.Println(string(t))
	_, err = c.CallCUDAFuncRetInt("cudaFree", a)
	if err != nil {
		panic(err)
	}
}
