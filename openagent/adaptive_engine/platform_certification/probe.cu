#include <cstdio>
#include <cstdlib>
#ifdef OA_HIP
#include <hip/hip_runtime.h>
#define API(name) hip##name
#define SUCCESS hipSuccess
#else
#include <cuda_runtime.h>
#define API(name) cuda##name
#define SUCCESS cudaSuccess
#endif
__global__ void certify(int *output) { output[0] = 734; }
int main(int argc, char **argv) {
    int device = argc > 1 ? std::atoi(argv[1]) : 0;
    int *output = nullptr, host = 0;
    int count = 0;
    if (API(GetDeviceCount)(&count) != SUCCESS || device < 0 || device >= count) {
        std::puts("OPENAGENT_RUNTIME_UNAVAILABLE {\"reason\":\"Native CUDA/HIP runtime or requested device unavailable\"}");
        return 2;
    }
    if (API(SetDevice)(device) != SUCCESS) return 1;
    if (API(Malloc)((void **)&output, sizeof(int)) != SUCCESS) return 1;
    certify<<<1, 1>>>(output);
    bool ok = API(GetLastError)() == SUCCESS && API(DeviceSynchronize)() == SUCCESS &&
        API(Memcpy)(&host, output, sizeof(int), API(MemcpyDeviceToHost)) == SUCCESS;
    if (API(Free)(output) != SUCCESS) ok = false;
    if (!ok || host != 734) return 2;
    std::puts("OPENAGENT_NATIVE_EXECUTED");
    return 0;
}
