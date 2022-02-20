package cudago

type CUDAMemcpyKind int

const (
	CUDAMemcpyHostToHost     = CUDAMemcpyKind(0) /**< Host   -> Host */
	CUDAMemcpyHostToDevice   = CUDAMemcpyKind(1) /**< Host   -> Device */
	CUDAMemcpyDeviceToHost   = CUDAMemcpyKind(2) /**< Device -> Host */
	CUDAMemcpyDeviceToDevice = CUDAMemcpyKind(3) /**< Device -> Device */
	CUDAMemcpyDefault        = CUDAMemcpyKind(4) /**< Direction of the traμsfer is inferred from the pointer values. Requires unified virtual addressing */
)

type CudaLimit int

const (
	CudaLimitStackSize                    = CudaLimit(0x00) /**< GPU thread stack size */
	CudaLimitPrintfFifoSize               = CudaLimit(0x01) /**< GPU printf FIFO size */
	CudaLimitMallocHeapSize               = CudaLimit(0x02) /**< GPU malloc heap size */
	CudaLimitDevRuntimeSyncDepth          = CudaLimit(0x03) /**< GPU device runtime synchronize depth */
	CudaLimitDevRuntimePendingLaunchCount = CudaLimit(0x04) /**< GPU device runtime pending launch count */
	CudaLimitMaxL2FetchGranularity        = CudaLimit(0x05) /**< A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint */
)
