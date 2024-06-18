#include <stdio.h>
#include <cuda_runtime.h>
int _ConvertSMVer2Cores(int major, int minor);


int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        fprintf(stderr,"Device %d: %s\n", dev, deviceProp.name);
        fprintf(stderr,"  Total number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
        fprintf(stderr,"  Total number of CUDA cores: %d\n",
               deviceProp.multiProcessorCount * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
        fprintf(stderr,"  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        fprintf(stderr,"  Max threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        fprintf(stderr,"  Max block dimensions: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        fprintf(stderr,"  Max grid dimensions: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    }

    return 0;
}

int _ConvertSMVer2Cores(int major, int minor) {
    // Defines the number of CUDA cores per SM for each compute capability
    typedef struct {
        int SM;
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192},
        {0x50, 128}, {0x52, 128}, {0x53, 128},
        {0x60, 64},  {0x61, 128}, {0x62, 128}, {0x70, 64},
        {0x72, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128},
        {-1, -1}
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    fprintf(stderr,"MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
    return -1;
}
