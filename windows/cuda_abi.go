package windows

import (
	"fmt"
	"os"
	"path"
	"reflect"
	"syscall"
	"unsafe"

	"github.com/KarKLi/cudago"
)

// CUDAError_t CUDA error type
type CUDAError_t int

type cudaWindowsCall struct {
	d *syscall.DLL
}

// NewCUDAWindowsCall returns a CudaCall interface with loading CUDA dynamic library automatically.
//
// If failed, NewCUDAWindowsCall returns nil or panic.
func NewCUDAWindowsCall(libraryPath string) cudago.CudaCall {
	// find library
	if len(libraryPath) == 0 {
		if c := os.Getenv("CUDA_PATH"); len(c) == 0 {
			return nil
		} else {
			libraryPath = path.Join(c, "bin/cudart64_100.dll")
		}
	}
	d := syscall.MustLoadDLL(libraryPath)
	return newCUDAWindowsCall(d)
}

// newCUDAWindowsCall returns a CudaCall interface with loaded CUDA dynamic library
func newCUDAWindowsCall(d *syscall.DLL) cudago.CudaCall {
	caller := &cudaWindowsCall{
		d: d,
	}
	var runtimeVersion int
	r1, err := caller.CallCUDAFuncRetInt("cudaRuntimeGetVersion", &runtimeVersion)
	if err != nil {
		panic(err)
	}
	if r1 != 0 {
		panic(CUDAErrorHandler(caller, CUDAError_t(r1)))
	}
	if runtimeVersion != cudago.Version() {
		panic(fmt.Sprintf("cuda version not match, wants: %d, library returns: %d", cudago.Version(), runtimeVersion))
	}
	return caller
}

// CallCUDAFuncRetInt Call CUDA function with return value of type int (equals to C type int or enum)
func (c *cudaWindowsCall) CallCUDAFuncRetInt(funcName string, p ...interface{}) (r int, err error) {
	proc := c.d.MustFindProc(funcName)
	callP := make([]uintptr, 0, len(p))
	// use reflect for type
	for _, t := range p {
		switch reflect.TypeOf(t).Kind() {
		case reflect.Int, reflect.Int64:
			callP = append(callP, uintptr(reflect.ValueOf(t).Int()))
		case reflect.Uint, reflect.Uint64, reflect.Uintptr:
			callP = append(callP, uintptr(reflect.ValueOf(t).Uint()))
		case reflect.Float64:
			callP = append(callP, uintptr(reflect.ValueOf(t).Float()))
		case reflect.Slice, reflect.Ptr, reflect.UnsafePointer:
			callP = append(callP, reflect.ValueOf(t).Pointer())
		case reflect.String:
			// convert it into *byte
			b, err := syscall.BytePtrFromString(reflect.ValueOf(t).String())
			if err != nil {
				return 0, fmt.Errorf("not valid string %v", t)
			}
			callP = append(callP, uintptr(unsafe.Pointer(b)))
		default:
			// We don't support bool, complex, array, chan, func, interface, map, struct for compatiablity.
			// bool is not support for C language.
			// and struct will cause unknown variable length, causing stack crash.
			return 0, fmt.Errorf("not supported type: %v", reflect.TypeOf(t).Kind())
		}
	}
	r1, _, errno := proc.Call(callP...)
	if errno != syscall.Errno(0) {
		return 0, fmt.Errorf("errno %d", errno)
	}
	// 将其解析成int返回
	return int(r1), nil
}

// CallCUDAFuncRetString Call CUDA function with return value of type string (equals to C type char * or const char *)
func (c *cudaWindowsCall) CallCUDAFuncRetString(funcName string, p ...interface{}) (r string, err error) {
	proc := c.d.MustFindProc(funcName)
	callP := make([]uintptr, 0, len(p))
	// use reflect for type
	for _, t := range p {
		switch reflect.TypeOf(t).Kind() {
		case reflect.Int, reflect.Int64:
			callP = append(callP, uintptr(reflect.ValueOf(t).Int()))
		case reflect.Uint, reflect.Uint64, reflect.Uintptr:
			callP = append(callP, uintptr(reflect.ValueOf(t).Uint()))
		case reflect.Float64:
			callP = append(callP, uintptr(reflect.ValueOf(t).Float()))
		case reflect.Slice, reflect.Ptr, reflect.UnsafePointer:
			callP = append(callP, reflect.ValueOf(t).Pointer())
		case reflect.String:
			// convert it into *byte
			b, err := syscall.BytePtrFromString(reflect.ValueOf(t).String())
			if err != nil {
				return "", fmt.Errorf("not valid string %v", t)
			}
			callP = append(callP, uintptr(unsafe.Pointer(b)))
		default:
			// We don't support bool, complex, array, chan, func, interface, map, struct for compatiablity.
			// bool is not support for C language.
			// and struct will cause unknown variable length, causing stack crash.
			return "", fmt.Errorf("not supported type: %v", reflect.TypeOf(t).Kind())
		}
	}
	r1, _, errno := proc.Call(callP...)
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
