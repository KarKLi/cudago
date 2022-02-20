package windows

import (
	"fmt"
	"reflect"
	"syscall"
	"unsafe"

	"github.com/karkli/cudago"
)

// CUDAError_t CUDA error type
type CUDAError_t int

type cudaWindowsCall struct {
	d *syscall.DLL
}

// NewCUDAWindowsCall returns a CudaCall interface with loaded CUDA dynamic library
func NewCUDAWindowsCall(d *syscall.DLL) cudago.CudaCall {
	// 先确认当前传入的d是否加载的10.0版本的cuda dll，否则报错
	caller := &cudaWindowsCall{
		d: d,
	}
	var runtimeVersion int
	r1, err := caller.CallCUDAFuncRetInt("cudaRuntimeGetVersion", &runtimeVersion)
	if err != nil {
		panic(err)
	}
	if r1 != 0 {
		panic(cudaErrorHandler(caller, CUDAError_t(r1)))
	}
	if runtimeVersion != cudago.Version() {
		panic(fmt.Sprintf("cuda version not match, wants: %d, library returns: %d", cudago.Version(), runtimeVersion))
	}
	return caller
}

// CallCUDAFuncRetInt Call CUDA function with return value of type int (equals to C type int or enum)
func (c *cudaWindowsCall) CallCUDAFuncRetInt(funcName string, p ...interface{}) (r int, err error) {
	proc := c.d.MustFindProc(funcName)
	var callP []uintptr
	// use reflect for type
	for _, t := range p {
		switch reflect.TypeOf(t).Kind() {
		case reflect.Int, reflect.Int64:
			callP = append(callP, uintptr(t.(int64)))
		case reflect.Uint, reflect.Uint64:
			callP = append(callP, uintptr(t.(uint64)))
		case reflect.Uintptr:
			callP = append(callP, t.(uintptr))
		case reflect.Float64:
			callP = append(callP, uintptr(t.(float64)))
		case reflect.Slice, reflect.Ptr:
			callP = append(callP, reflect.ValueOf(t).Pointer())
		case reflect.String:
			// convert it into *byte
			b, err := syscall.BytePtrFromString(t.(string))
			if err != nil {
				return 0, fmt.Errorf("not valid string %v", t)
			}
			callP = append(callP, uintptr(unsafe.Pointer(b)))
		case reflect.Struct, reflect.UnsafePointer:
			callP = append(callP, reflect.ValueOf(t).UnsafeAddr())
		default:
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
	var callP []uintptr
	// use reflect for type
	for _, t := range p {
		switch reflect.TypeOf(t).Kind() {
		case reflect.Int, reflect.Int64:
			callP = append(callP, uintptr(t.(int64)))
		case reflect.Uint, reflect.Uint64:
			callP = append(callP, uintptr(t.(uint64)))
		case reflect.Uintptr:
			callP = append(callP, t.(uintptr))
		case reflect.Float64:
			callP = append(callP, uintptr(t.(float64)))
		case reflect.Slice, reflect.Ptr:
			callP = append(callP, reflect.ValueOf(t).Pointer())
		case reflect.String:
			// convert it into *byte
			b, err := syscall.BytePtrFromString(t.(string))
			if err != nil {
				return "", fmt.Errorf("not valid string %v", t)
			}
			callP = append(callP, uintptr(unsafe.Pointer(b)))
		case reflect.Struct, reflect.UnsafePointer:
			callP = append(callP, reflect.ValueOf(t).UnsafeAddr())
		default:
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
