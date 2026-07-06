#include "cuda_utils.h"

int main(int argc, char** argv) {
    int numDevices = 0;    
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    for (int i=0; i<numDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaInitDevice(0, 0, 0));

        int deviceId = i;
        int concurrentManagedAccess = -1;
        // cudaDevAttrConcurrentManagedAccess-统一内存支持:
        //      1-完整支持
        //      0-有限支持 
        CUDA_CHECK(cudaDeviceGetAttribute(&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, deviceId));
        
        int pageableMemoryAccess = -1;
        // cudaDevAttrPageableMemoryAccess-哪些系统内存是完全受支持的统一内存:
        //      1-所有系统内存
        //      0-只有显示分配为管理内存的内存
        CUDA_CHECK(cudaDeviceGetAttribute(&pageableMemoryAccess, cudaDevAttrPageableMemoryAccess, deviceId));

        int pageableMemoryAccessUsesHostPageTables = -1;
        // cudaDevAttrPageableMemoryAccessUsesHostPageTables-CPU/GPU一致性机制
        //      1-硬件
        //      0-软件
        CUDA_CHECK(cudaDeviceGetAttribute(&pageableMemoryAccessUsesHostPageTables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, deviceId));
        printf("Device %d has: ", deviceId);
        if (concurrentManagedAccess) {
            if (pageableMemoryAccess) {
                printf("full unified memory support ");
                if (pageableMemoryAccessUsesHostPageTables) {
                    printf(" with hardware coherency \n");
                } else {
                    printf(" with software coherency \n");
                }
            } else {
                printf("full unified memory support for CUDA_made managed allocations \n");
            }
        } else {
            printf("limited unified memory support: Windows, WSL, or Tegra \n");
        }
    }
    return 0;
}