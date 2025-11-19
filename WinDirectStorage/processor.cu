#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <iomanip>
#include <vector>
#include <algorithm>

// Windows & D3D12
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <comdef.h>

// DirectStorage
#include <dstorage.h>

using Microsoft::WRL::ComPtr;

// =================================================================================
// CONFIGURATION & STRUCTS
// =================================================================================

#define MAX_NAME_LEN 100
#define GLOBAL_HASH_SIZE 65536 
#define SHM_HASH_SIZE 1024 

// FNV-1a 64-bit Constants
#define FNV_PRIME_64 1099511628211ULL
#define FNV_OFFSET_BASIS_64 14695981039346656037ULL

// IO Configuration
const int NUM_STREAM_BUFFERS = 3;
const uint64_t STREAM_CHUNK_SIZE = 32000000; // 32 MB chunks
const size_t OVERLAP_SIZE = 65536; // 64KB overlap
const int THREADS_PER_BLOCK = 256;

// CPU-Side Helper
struct StationStats {
    int min;
    int max;
    long long sum;
    int count;
};

// GPU-Side Stats (16 bytes aligned)
struct __align__(16) CompactStats {
    long long sum;
    int min;
    int max;
    int count;
};

// Global Map Entry
struct GlobalMapEntry {
    char name[MAX_NAME_LEN];
    CompactStats stats;
    unsigned long long hash; // 64-bit Hash
};

// Shared Memory Entry
struct ShmEntry {
    unsigned long long hash;
    int name_offset;
    CompactStats stats;
};

struct StreamBuffer {
    ComPtr<ID3D12Resource> d3d12Buffer;
    ComPtr<ID3D12Fence> d3dFence;
    UINT64 fenceValue;
    HANDLE fenceEvent;

    cudaExternalMemory_t cudaExtMem;
    void* cudaPtr;
    cudaStream_t cudaStream;
    cudaEvent_t processingComplete;

    bool inUse;
    uint64_t allocationSize; // Store the real allocation size
};

// =================================================================================
// HELPER FUNCTIONS
// =================================================================================

void ThrowIfFailed(HRESULT hr, const char* msg) {
    if (FAILED(hr)) {
        _com_error err(hr);
        std::cerr << "HRESULT Error: " << msg << " (0x" << std::hex << hr << ")" << std::endl;
        throw std::runtime_error(msg);
    }
}

void ThrowIfCudaFailed(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        throw std::runtime_error(msg);
    }
}

// =================================================================================
// CUDA KERNELS
// =================================================================================

__device__ inline void atomic_update_stats(CompactStats* target, int val) {
    atomicAdd((unsigned long long*) & target->sum, (unsigned long long)(long long)val);
    atomicMin(&target->min, val);
    atomicMax(&target->max, val);
    atomicAdd(&target->count, 1);
}

__global__ void parse_and_aggregate_kernel(
    const char* __restrict__ data,
    size_t chunk_size,
    size_t buffer_limit,
    uint64_t chunk_offset_global,
    GlobalMapEntry* global_map)
{
    __shared__ ShmEntry shm_entries[SHM_HASH_SIZE];

    size_t tid = threadIdx.x;
    for (int i = tid; i < SHM_HASH_SIZE; i += blockDim.x) {
        shm_entries[i].hash = 0;
        shm_entries[i].stats.min = INT_MAX;
        shm_entries[i].stats.max = INT_MIN;
        shm_entries[i].stats.sum = 0;
        shm_entries[i].stats.count = 0;
    }
    __syncthreads();

    size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t WINDOW_SIZE = 128;

    for (size_t offset = idx * WINDOW_SIZE; offset < buffer_limit; offset += stride * WINDOW_SIZE) {
        const char* ptr = data + offset;
        const char* end_window = data + min(offset + WINDOW_SIZE, buffer_limit); 
        const char* buffer_end = data + buffer_limit; // Absolute end of valid data

        if (offset == 0) {
            if (chunk_offset_global > 0) {
                 while (ptr < buffer_end && *ptr != '\n') ptr++;
                 ptr++;
            }
        }
        else {
            // Skip if we are NOT at the start of a line
            if (offset > 0 && *(ptr - 1) != '\n') {
                while (ptr < buffer_end && *ptr != '\n') ptr++;
                ptr++;
            }
        }

        // Main Processing Loop
        while (ptr < end_window) {
            
            if ((size_t)(ptr - data) > chunk_size) break;
            if (ptr >= buffer_end) break;

            const char* line_start = ptr;
            unsigned long long hash = FNV_OFFSET_BASIS_64;

            while (*ptr != ';') {
                hash = (hash ^ (unsigned char)*ptr) * FNV_PRIME_64;
                ptr++;
            }
            ptr++;

            int sign = 1;
            if (*ptr == '-') { sign = -1; ptr++; }

            int temp_val = 0;
            if (*(ptr + 1) == '.') {
                temp_val = (*ptr - '0') * 10 + (*(ptr + 2) - '0');
                ptr += 4;
            }
            else {
                temp_val = (*ptr - '0') * 100 + (*(ptr + 1) - '0') * 10 + (*(ptr + 3) - '0');
                ptr += 5;
            }
            temp_val *= sign;

            int slot = hash & (SHM_HASH_SIZE - 1);
            bool inserted = false;
            while (!inserted) {
                unsigned long long old = atomicCAS(&shm_entries[slot].hash, 0ULL, hash);
                if (old == 0ULL || old == hash) {
                    if (old == 0ULL) shm_entries[slot].name_offset = (int)(line_start - data);
                    atomic_update_stats(&shm_entries[slot].stats, temp_val);
                    inserted = true;
                }
                else {
                    slot = (slot + 1) & (SHM_HASH_SIZE - 1);
                }
            }
        }
    }
    __syncthreads();

    for (int i = tid; i < SHM_HASH_SIZE; i += blockDim.x) {
        if (shm_entries[i].hash != 0) {
            unsigned long long hash = shm_entries[i].hash;
            int slot = hash % GLOBAL_HASH_SIZE;

            while (true) {
                unsigned long long old = atomicCAS(&global_map[slot].hash, 0ULL, hash);

                if (old == 0ULL || old == hash) {
                    if (old == 0ULL) {
                        const char* src = data + shm_entries[i].name_offset;
                        int c_idx = 0;
                        while (src[c_idx] != ';' && c_idx < MAX_NAME_LEN - 1) {
                            global_map[slot].name[c_idx] = src[c_idx];
                            c_idx++;
                        }
                        global_map[slot].name[c_idx] = '\0';
                    }

                    atomicAdd((unsigned long long*) & global_map[slot].stats.sum, (unsigned long long)shm_entries[i].stats.sum);
                    atomicMin(&global_map[slot].stats.min, shm_entries[i].stats.min);
                    atomicMax(&global_map[slot].stats.max, shm_entries[i].stats.max);
                    atomicAdd(&global_map[slot].stats.count, shm_entries[i].stats.count);

                    break;
                }
                slot = (slot + 1) % GLOBAL_HASH_SIZE;
            }
        }
    }
}

__global__ void init_global_map_kernel(GlobalMapEntry* entries, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        entries[idx].hash = 0; // 0 indicates empty
        entries[idx].stats.min = INT_MAX;
        entries[idx].stats.max = INT_MIN;
        entries[idx].stats.sum = 0;
        entries[idx].stats.count = 0;
        // Name doesn't need clearing if we check hash, 
        // but safety first:
        entries[idx].name[0] = '\0'; 
    }
}

// =================================================================================
// MAIN
// =================================================================================

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    const char* inputFileName = argv[1];
    const char* outputFileName = argv[2];

    LARGE_INTEGER freq, start_total, end_total;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start_total);

    ComPtr<ID3D12Device> pDevice;
    ComPtr<IDXGIFactory4> pDxgiFactory;
    ComPtr<IDStorageFactory> pDsFactory;
    ComPtr<IDStorageFile> pDsFile;
    ComPtr<IDStorageQueue> pDsQueue;

    GlobalMapEntry* d_globalMap = nullptr;
    StreamBuffer streamBuffers[NUM_STREAM_BUFFERS];

    try {
        // Init D3D12
        CreateDXGIFactory2(0, IID_PPV_ARGS(&pDxgiFactory));
        ComPtr<IDXGIAdapter1> pAdapter;
        pDxgiFactory->EnumAdapters1(0, &pAdapter);
        D3D12CreateDevice(pAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pDevice));

        // CUDA Setup: Find device matching the DXGI Adapter
        DXGI_ADAPTER_DESC1 adapterDesc;
        pAdapter->GetDesc1(&adapterDesc);
        int numCudaDevices = 0;
        cudaGetDeviceCount(&numCudaDevices);
        int cudaDeviceID = -1;
        for (int i = 0; i < numCudaDevices; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            if (memcmp(prop.luid, &adapterDesc.AdapterLuid, sizeof(adapterDesc.AdapterLuid)) == 0) {
                cudaDeviceID = i;
                break;
            }
        }
        if (cudaDeviceID == -1) {
            cudaDeviceID = 0;
            printf("Warning: Could not match LUID, defaulting to Device 0\n");
        }
        cudaSetDevice(cudaDeviceID);

        // Create a DirectStorage factory and open the file
        DStorageGetFactory(IID_PPV_ARGS(&pDsFactory));
        std::wstring wFileName(inputFileName, inputFileName + strlen(inputFileName));
        ThrowIfFailed(pDsFactory->OpenFile(wFileName.c_str(), IID_PPV_ARGS(&pDsFile)), "OpenFile");

        BY_HANDLE_FILE_INFORMATION fileInfo = {};
        pDsFile->GetFileInformation(&fileInfo);
        uint64_t fileSize = (static_cast<uint64_t>(fileInfo.nFileSizeHigh) << 32) | fileInfo.nFileSizeLow;

        DSTORAGE_QUEUE_DESC dsQueueDesc = {};
        dsQueueDesc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
        dsQueueDesc.Priority = DSTORAGE_PRIORITY_HIGH;
        dsQueueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        dsQueueDesc.Device = pDevice.Get();
    
        pDsFactory->CreateQueue(&dsQueueDesc, IID_PPV_ARGS(&pDsQueue));

        // Create stream buffers
        for (int i = 0; i < NUM_STREAM_BUFFERS; i++) {
            auto& buf = streamBuffers[i];

            // Define D3D12 resource
            D3D12_RESOURCE_DESC resDesc = {};
            resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            resDesc.Width = STREAM_CHUNK_SIZE + OVERLAP_SIZE;
            resDesc.Height = 1;
            resDesc.DepthOrArraySize = 1;
            resDesc.MipLevels = 1;
            resDesc.SampleDesc.Count = 1;
            resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            resDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

            // Store real allocation size for CUDA
            D3D12_RESOURCE_ALLOCATION_INFO allocInfo = pDevice->GetResourceAllocationInfo(0, 1, &resDesc);
            buf.allocationSize = allocInfo.SizeInBytes;

            D3D12_HEAP_PROPERTIES heapProps = { D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1 };

            ThrowIfFailed(pDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_SHARED, &resDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buf.d3d12Buffer)), "Create buffer");

            pDevice->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&buf.d3dFence));
            buf.fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
            buf.fenceValue = 0;

            // Create Shared Handle
            HANDLE sharedHandle = nullptr;
            ThrowIfFailed(pDevice->CreateSharedHandle(buf.d3d12Buffer.Get(), nullptr, GENERIC_ALL, nullptr, &sharedHandle), "CreateSharedHandle");

            // Import D3D12-allocated memory into CUDA using the physical allocation size
            cudaExternalMemoryHandleDesc extDesc = {};
            extDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
            extDesc.size = buf.allocationSize;
            extDesc.handle.win32.handle = sharedHandle;
            extDesc.flags = cudaExternalMemoryDedicated;

            ThrowIfCudaFailed(cudaImportExternalMemory(&buf.cudaExtMem, &extDesc), "Import memory");
            CloseHandle(sharedHandle);

            cudaExternalMemoryBufferDesc bufDesc = {};
            bufDesc.size = STREAM_CHUNK_SIZE + OVERLAP_SIZE; // We still map the logical size we need
            bufDesc.offset = 0;
            ThrowIfCudaFailed(cudaExternalMemoryGetMappedBuffer(&buf.cudaPtr, buf.cudaExtMem, &bufDesc), "Map buffer");

            cudaStreamCreate(&buf.cudaStream);
            cudaEventCreate(&buf.processingComplete);
            buf.inUse = false;
        }

        // Global Map Alloc
        ThrowIfCudaFailed(cudaMalloc(&d_globalMap, GLOBAL_HASH_SIZE * sizeof(GlobalMapEntry)), "Malloc map");

        // Launch global map:
        int initBlockSize = 256;
        int initGridSize = (GLOBAL_HASH_SIZE + initBlockSize - 1) / initBlockSize;
        init_global_map_kernel<<<initGridSize, initBlockSize>>>(d_globalMap, GLOBAL_HASH_SIZE);
        ThrowIfCudaFailed(cudaGetLastError(), "Init kernel launch");
        ThrowIfCudaFailed(cudaDeviceSynchronize(), "Init kernel sync");

        // PIPELINED EXECUTION
        uint64_t currentFileOffset = 0;
        int currentBuffer = 0;
        UINT64 nextFenceValue = 1;

        while (currentFileOffset < fileSize) {
            StreamBuffer* buf = &streamBuffers[currentBuffer];

            if (buf->inUse) {
                cudaEventSynchronize(buf->processingComplete);
                buf->inUse = false;
            }

            uint64_t readSize = min(STREAM_CHUNK_SIZE + OVERLAP_SIZE, fileSize - currentFileOffset);

            // Submit IO
            DSTORAGE_REQUEST request = {};
            request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
            request.Source.File.Source = pDsFile.Get();
            request.Source.File.Offset = currentFileOffset;
            request.Source.File.Size = (uint32_t)readSize;
            request.UncompressedSize = (uint32_t)readSize;
            request.Destination.Buffer.Resource = buf->d3d12Buffer.Get();
            request.Destination.Buffer.Offset = 0;
            request.Destination.Buffer.Size = (uint32_t)readSize;

            pDsQueue->EnqueueRequest(&request);
            buf->fenceValue = nextFenceValue++;
            pDsQueue->EnqueueSignal(buf->d3dFence.Get(), buf->fenceValue);
            pDsQueue->Submit();
            buf->inUse = true;

            // Wait for IO
            buf->d3dFence->SetEventOnCompletion(buf->fenceValue, buf->fenceEvent);
            WaitForSingleObject(buf->fenceEvent, INFINITE);

            int num_blocks = min(GLOBAL_HASH_SIZE, (int)((readSize + 128 * THREADS_PER_BLOCK - 1) / (128 * THREADS_PER_BLOCK)));

            parse_and_aggregate_kernel << <num_blocks, THREADS_PER_BLOCK, 0, buf->cudaStream >> > (
                (char*)buf->cudaPtr,
                STREAM_CHUNK_SIZE,
                readSize,
                currentFileOffset,
                d_globalMap
                );

            cudaEventRecord(buf->processingComplete, buf->cudaStream);

            currentFileOffset += STREAM_CHUNK_SIZE;
            currentBuffer = (currentBuffer + 1) % NUM_STREAM_BUFFERS;
        }

        for (int i = 0; i < NUM_STREAM_BUFFERS; i++) cudaEventSynchronize(streamBuffers[i].processingComplete);

        QueryPerformanceCounter(&end_total);

        // Readback
        GlobalMapEntry* h_map = (GlobalMapEntry*)malloc(GLOBAL_HASH_SIZE * sizeof(GlobalMapEntry));
        cudaMemcpy(h_map, d_globalMap, GLOBAL_HASH_SIZE * sizeof(GlobalMapEntry), cudaMemcpyDeviceToHost);

        std::map<std::string, StationStats> sortedStats;
        for (int i = 0; i < GLOBAL_HASH_SIZE; i++) {
            if (h_map[i].hash != 0) {
                if (h_map[i].name[0] == 0) continue;

                std::string name(h_map[i].name);
                StationStats s;
                s.min = h_map[i].stats.min;
                s.max = h_map[i].stats.max;
                s.sum = h_map[i].stats.sum;
                s.count = h_map[i].stats.count;
                sortedStats[name] = s;
            }
        }

        std::ofstream outfile(outputFileName);
        outfile << "{";
        bool first = true;
        for (const auto& kv : sortedStats) {
            if (!first) outfile << ", ";
            double minVal = kv.second.min / 10.0;
            double avgVal = (kv.second.sum / 10.0) / kv.second.count;
            double maxVal = kv.second.max / 10.0;
            outfile << kv.first << "=" << std::fixed << std::setprecision(1) << minVal << "/" << avgVal << "/" << maxVal;
            first = false;
        }
        outfile << "}\n";
        outfile.close();

        free(h_map);
        cudaFree(d_globalMap);

        double time_ms = (double)(end_total.QuadPart - start_total.QuadPart) * 1000.0 / freq.QuadPart;
        printf("Total Time: %.2fms\n", time_ms);

    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}