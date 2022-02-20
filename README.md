# Go package cudago

<img src="https://go.dev/images/gophers/pilot-bust.svg" alt="go mascot" style="zoom:30%;" />

See [CUDA official page](https://developer.nvidia.cn/zh-cn/cuda-toolkit) for CUDA library info and doc.

cudago use dynamic library function calls for Go developers run CUDA with Go's other features.

cudago provides Windows and Linux platform implementation for CUDA users.

**Notice: cudago only support 64-bit system!**

Support OS and arch are shown at below table:

| OS/arch       | supported?         |
| ------------- | ------------------ |
| Windows/amd64 | support            |
| Linux/amd64   | under construction |
| Windows/arm64 | not test           |
| Linux/amd64   | not test           |

Current Main branch support for CUDA v10.0.

*karkli, mail: <karkli@tencent.com>*
