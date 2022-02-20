package cudago

// Version returns library match CUDA version
func Version() int {
	return 10000 /* match to CUDART_VERSION */
}
