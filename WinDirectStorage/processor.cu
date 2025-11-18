#include <cuda_runtime.h>
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

 // Windows & D3D12
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <comdef.h>

// DirectStorage
#include <dstorage.h>

using Microsoft::WRL::ComPtr;

// =================================================================================
// 1. CONFIGURATION & STRUCTS
// =================================================================================

#define MAX_NAME_LEN 100
#define HASH_SIZE 16384
#define LOCAL_HASH_SIZE 64
#define BLOCK_SIZE 256

const uint64_t DS_CHUNK_SIZE = 32 * 1024 * 1024; // 32MB chunks for DirectStorage

struct StationStats {
    char name[MAX_NAME_LEN];
    int min;
    int max;
    int64_t sum;
    int count;
    unsigned int hash;
    size_t name_len;
};

// =================================================================================
// 2. DEVICE HELPER FUNCTIONS (From your working code)
// =================================================================================

__device__ __host__ unsigned int hash_string(const char* str, size_t len) {
    unsigned int hash = 5381;
    for (size_t i = 0; i < len; i++) {
        hash = ((hash << 5) + hash) + str[i];
    }
    return hash;
}

__device__ void parse_line_int(const char* line, size_t line_len,
    char* station, size_t* station_len, int* temp_int) {
    size_t i = 0;
    // Find the separator
    while (i < line_len && line[i] != ';') {
        station[i] = line[i];
        i++;
    }
    *station_len = i;
    station[i] = '\0';

    i++; // Skip ';'
    if (i >= line_len) return;

    int sign = 1;
    int temp_val = 0;

    if (line[i] == '-') {
        sign = -1;
        i++;
    }

    // Fast, fixed-point parsing
    if (i + 1 < line_len && line[i + 1] == '.') { // Format: X.X or -X.X
        temp_val = (line[i] - '0') * 10 + (line[i + 2] - '0');
    }
    else if (i + 2 < line_len && line[i + 2] == '.') { // Format: XX.X or -XX.X
        temp_val = (line[i] - '0') * 100 + (line[i + 1] - '0') * 10 + (line[i + 3] - '0');
    }

    *temp_int = temp_val * sign;
}

__device__ bool strings_equal(const char* s1, const char* s2, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (s1[i] != s2[i]) return false;
    }
    return true;
}

// =================================================================================
// 3. CUDA KERNELS (From your working code)
// =================================================================================

__global__ void process_measurements_local(const char* data, const size_t* line_offsets, size_t num_lines,
    StationStats* thread_results, int* thread_counts) {
    size_t global_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t thread_id = global_idx;

    StationStats local_hash[LOCAL_HASH_SIZE];
    for (int i = 0; i < LOCAL_HASH_SIZE; i++) {
        local_hash[i].count = 0;
    }

    for (size_t idx = global_idx; idx < num_lines; idx += (size_t)blockDim.x * gridDim.x) {
        size_t start = line_offsets[idx];
        size_t end = line_offsets[idx + 1];
        size_t line_len = end - start;

        // Trim newline
        if (line_len > 0 && data[start + line_len - 1] == '\n') line_len--;
        if (line_len > 0 && data[start + line_len - 1] == '\r') line_len--;

        if (line_len <= 1) continue;

        char station[MAX_NAME_LEN];
        size_t station_len;
        int temp_int;

        parse_line_int(data + start, line_len, station, &station_len, &temp_int);

        if (station_len == 0) continue;

        unsigned int hash = hash_string(station, station_len);
        int slot = hash % LOCAL_HASH_SIZE;

        bool inserted = false;
        for (int probe = 0; probe < LOCAL_HASH_SIZE && !inserted; probe++) {
            int current_slot = (slot + probe) % LOCAL_HASH_SIZE;

            if (local_hash[current_slot].count == 0) {
                for (size_t i = 0; i < station_len; i++) {
                    local_hash[current_slot].name[i] = station[i];
                }
                local_hash[current_slot].name[station_len] = '\0';
                local_hash[current_slot].min = temp_int;
                local_hash[current_slot].max = temp_int;
                local_hash[current_slot].sum = (int64_t)temp_int;
                local_hash[current_slot].count = 1;
                local_hash[current_slot].hash = hash;
                local_hash[current_slot].name_len = station_len;
                inserted = true;
            }
            else if (local_hash[current_slot].hash == hash &&
                local_hash[current_slot].name_len == station_len &&
                strings_equal(local_hash[current_slot].name, station, station_len)) {
                if (temp_int < local_hash[current_slot].min) local_hash[current_slot].min = temp_int;
                if (temp_int > local_hash[current_slot].max) local_hash[current_slot].max = temp_int;
                local_hash[current_slot].sum += (int64_t)temp_int;
                local_hash[current_slot].count++;
                inserted = true;
            }
        }
    }

    int write_idx = 0;
    for (int i = 0; i < LOCAL_HASH_SIZE; i++) {
        if (local_hash[i].count > 0) {
            thread_results[thread_id * LOCAL_HASH_SIZE + write_idx] = local_hash[i];
            write_idx++;
        }
    }
    thread_counts[thread_id] = write_idx;
}

__global__ void merge_results(StationStats* thread_results, int* thread_counts, int num_threads,
    StationStats* global_hash) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_threads) return;

    int count = thread_counts[idx];
    for (int i = 0; i < count; i++) {
        StationStats* stat = &thread_results[idx * LOCAL_HASH_SIZE + i];
        unsigned int hash = stat->hash;
        int slot = hash % HASH_SIZE;

        bool inserted = false;
        for (int probe = 0; probe < HASH_SIZE && !inserted; probe++) {
            int current_slot = (slot + probe) % HASH_SIZE;

            int old_count = atomicCAS((int*)&global_hash[current_slot].count, 0, -1);

            if (old_count == 0) {
                size_t name_len = 0;
                while (stat->name[name_len] != '\0') name_len++;
                for (size_t j = 0; j < name_len; j++) {
                    global_hash[current_slot].name[j] = stat->name[j];
                }
                global_hash[current_slot].name[name_len] = '\0';

                global_hash[current_slot].min = stat->min;
                global_hash[current_slot].max = stat->max;
                global_hash[current_slot].sum = stat->sum;
                global_hash[current_slot].hash = stat->hash;

                __threadfence();

                global_hash[current_slot].count = stat->count;
                inserted = true;
            }
            else if (old_count != -1) {
                while (atomicAdd((int*)&global_hash[current_slot].count, 0) == -1);

                if (global_hash[current_slot].hash == hash) {
                    size_t name_len = 0;
                    while (stat->name[name_len] != '\0') name_len++;

                    if (strings_equal(global_hash[current_slot].name, stat->name, name_len)) {
                        atomicMin(&global_hash[current_slot].min, stat->min);
                        atomicMax(&global_hash[current_slot].max, stat->max);
                        atomicAdd((unsigned long long*) & global_hash[current_slot].sum,
                            (unsigned long long)stat->sum);
                        atomicAdd(&global_hash[current_slot].count, stat->count);
                        inserted = true;
                    }
                }
            }
        }
    }
}

// =================================================================================
// 4. GPU LINE OFFSET FINDER KERNELS
// =================================================================================

// Single-pass kernel: Each thread writes its newlines directly with atomic positioning
__global__ void build_offsets_kernel_atomic(const char* data, size_t data_size,
    size_t* offsets, size_t* counter) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t i = idx; i < data_size; i += stride) {
        if (data[i] == '\n' && i + 1 < data_size) {
            // Get unique position in offsets array
            size_t pos = atomicAdd((unsigned long long*)counter, 1ULL);
            // Write offset (+1 because offsets[0] = 0 is pre-set)
            offsets[pos + 1] = i + 1;
        }
    }
}

// Two-pass approach (more efficient): Count per-block, then write with known positions
__global__ void count_newlines_per_block(const char* data, size_t data_size,
    unsigned int* block_counts) {
    __shared__ unsigned int shared_count;

    if (threadIdx.x == 0) {
        shared_count = 0;
    }
    __syncthreads();

    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    unsigned int local_count = 0;
    for (size_t i = idx; i < data_size; i += stride) {
        if (data[i] == '\n') {
            local_count++;
        }
    }

    // Reduce within block
    atomicAdd(&shared_count, local_count);
    __syncthreads();

    // Write block total
    if (threadIdx.x == 0) {
        block_counts[blockIdx.x] = shared_count;
    }
}

__global__ void build_offsets_two_pass(const char* data, size_t data_size,
    size_t* offsets,
    const unsigned int* block_offsets) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate where this thread should start writing
    // Only process the data that belongs to this thread (no grid-stride)
    if (idx >= data_size) return;

    // Count newlines BEFORE this thread's starting position to know local offset
    unsigned int local_offset = 0;
    size_t thread_start = idx;

    // Get this block's write base
    size_t write_base = (blockIdx.x == 0) ? 0 : block_offsets[blockIdx.x - 1];

    // Count newlines from block start to this thread
    size_t block_start = (size_t)blockIdx.x * blockDim.x;
    for (size_t i = block_start; i < thread_start && i < data_size; i++) {
        if (data[i] == '\n') {
            local_offset++;
        }
    }

    // Now process this thread's single element
    if (data[idx] == '\n' && idx + 1 < data_size) {
        offsets[write_base + local_offset + 1] = idx + 1;
    }
}

// Simple inclusive prefix sum on CPU (for small block_counts array)
void inclusive_prefix_sum_cpu(unsigned int* data, size_t n, unsigned int* output) {
    if (n == 0) return;
    output[0] = data[0];
    for (size_t i = 1; i < n; i++) {
        output[i] = output[i - 1] + data[i];
    }
}

// =================================================================================
// 5. INFRASTRUCTURE HELPERS
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
// 6. MAIN
// =================================================================================

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    const char* inputFileName = argv[1];
    const char* outputFileName = argv[2];

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    // Timing helper lambda
    auto get_elapsed_ms = [&freq](LARGE_INTEGER start, LARGE_INTEGER end) -> double {
        return (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
        };

    LARGE_INTEGER start_total, end_total;
    LARGE_INTEGER start_io, end_io;
    LARGE_INTEGER start_offsets, end_offsets;
    LARGE_INTEGER start_compute, end_compute;
    LARGE_INTEGER start_readback, end_readback;

    QueryPerformanceCounter(&start_total);

    // D3D12 & DS Objects
    ComPtr<ID3D12Device> pDevice;
    ComPtr<IDXGIFactory4> pDxgiFactory;
    ComPtr<ID3D12CommandQueue> pCommandQueue;
    ComPtr<ID3D12CommandAllocator> pCommandAllocator;
    ComPtr<ID3D12GraphicsCommandList> pCommandList;
    ComPtr<ID3D12Fence> pFence;
    ComPtr<ID3D12Resource> pBuffer;
    ComPtr<ID3D12Resource> pReadbackBuffer;
    ComPtr<IDStorageFactory> pDsFactory;
    ComPtr<IDStorageFile> pDsFile;
    ComPtr<IDStorageQueue> pDsQueue;

    HANDLE fenceEvent = nullptr;
    UINT64 fenceValue = 1;

    // CUDA Objects
    cudaExternalMemory_t extMemFile = nullptr;
    void* d_fileData = nullptr;
    StationStats* d_globalHash = nullptr;
    size_t* d_lineOffsets = nullptr;
    StationStats* d_threadResults = nullptr;
    int* d_threadCounts = nullptr;

    try {
        // --- 1. Init D3D12 & CUDA ---
        printf("[Init] Initializing D3D12 and CUDA...\n");

        CreateDXGIFactory2(0, IID_PPV_ARGS(&pDxgiFactory));

        ComPtr<IDXGIAdapter1> pAdapter;
        pDxgiFactory->EnumAdapters1(0, &pAdapter);

        D3D12CreateDevice(pAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pDevice));

        // Find matching CUDA device
        int numCudaDevices = 0;
        cudaGetDeviceCount(&numCudaDevices);

        DXGI_ADAPTER_DESC1 adapterDesc;
        pAdapter->GetDesc1(&adapterDesc);

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
            throw std::runtime_error("No matching CUDA device found.");
        }

        ThrowIfCudaFailed(cudaSetDevice(cudaDeviceID), "cudaSetDevice");

        // --- 2. Init DirectStorage & Get File Info ---
        printf("[IO] Opening file: %s\n", inputFileName);

        DStorageGetFactory(IID_PPV_ARGS(&pDsFactory));

        std::wstring wFileName(inputFileName, inputFileName + strlen(inputFileName));
        ThrowIfFailed(pDsFactory->OpenFile(wFileName.c_str(), IID_PPV_ARGS(&pDsFile)), "OpenFile");

        BY_HANDLE_FILE_INFORMATION fileInfo = {};
        pDsFile->GetFileInformation(&fileInfo);
        uint64_t fileSize = (static_cast<uint64_t>(fileInfo.nFileSizeHigh) << 32) | fileInfo.nFileSizeLow;

        printf("     Size: %llu bytes (%.2f GB)\n", fileSize, fileSize / (1024.0 * 1024.0 * 1024.0));

        // --- 3. Create D3D12 Resources ---
        D3D12_COMMAND_QUEUE_DESC queueDesc = {
            D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0
        };
        pDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&pCommandQueue));
        pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&pCommandAllocator));
        pDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
            pCommandAllocator.Get(), nullptr, IID_PPV_ARGS(&pCommandList));
        pCommandList->Close();

        pDevice->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&pFence));
        fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

        // File Buffer (GPU)
        D3D12_HEAP_PROPERTIES heapProps = {
            D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            D3D12_MEMORY_POOL_UNKNOWN, 1, 1
        };

        D3D12_RESOURCE_DESC resDesc = {};
        resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resDesc.Width = fileSize;
        resDesc.Height = 1;
        resDesc.DepthOrArraySize = 1;
        resDesc.MipLevels = 1;
        resDesc.SampleDesc.Count = 1;
        resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        ThrowIfFailed(pDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_SHARED, &resDesc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&pBuffer)),
            "Create File Buffer");

        // Readback Buffer (for results)
        uint64_t resultSize = HASH_SIZE * sizeof(StationStats);
        heapProps.Type = D3D12_HEAP_TYPE_READBACK;
        resDesc.Width = resultSize;
        resDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

        ThrowIfFailed(pDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&pReadbackBuffer)),
            "Create Readback Buffer");

        // --- 4. DirectStorage Load ---
        printf("[IO] Loading file via DirectStorage...\n");
        QueryPerformanceCounter(&start_io);

        DSTORAGE_QUEUE_DESC dsQueueDesc = {};
        dsQueueDesc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
        dsQueueDesc.Priority = DSTORAGE_PRIORITY_NORMAL;
        dsQueueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        dsQueueDesc.Device = pDevice.Get();

        pDsFactory->CreateQueue(&dsQueueDesc, IID_PPV_ARGS(&pDsQueue));

        uint64_t numChunks = (fileSize + DS_CHUNK_SIZE - 1) / DS_CHUNK_SIZE;
        printf("     Enqueuing %llu read requests...\n", numChunks);

        for (uint64_t i = 0; i < numChunks; ++i) {
            DSTORAGE_REQUEST request = {};
            request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
            request.Source.File.Source = pDsFile.Get();

            uint64_t offset = i * DS_CHUNK_SIZE;
            uint32_t size = (uint32_t)((offset + DS_CHUNK_SIZE > fileSize) ?
                (fileSize - offset) : DS_CHUNK_SIZE);

            request.Source.File.Offset = offset;
            request.Source.File.Size = size;
            request.UncompressedSize = size;
            request.Destination.Buffer.Resource = pBuffer.Get();
            request.Destination.Buffer.Offset = offset;
            request.Destination.Buffer.Size = size;

            pDsQueue->EnqueueRequest(&request);
        }

        pDsQueue->EnqueueSignal(pFence.Get(), fenceValue);
        pDsQueue->Submit();

        printf("     Waiting for GPU load...\n");
        if (pFence->GetCompletedValue() < fenceValue) {
            pFence->SetEventOnCompletion(fenceValue, fenceEvent);
            WaitForSingleObject(fenceEvent, INFINITE);
        }
        fenceValue++;

        QueryPerformanceCounter(&end_io);
        printf("     DirectStorage load completed in %.2f ms\n", get_elapsed_ms(start_io, end_io));

        // --- 5. Import to CUDA ---
        printf("[CUDA] Importing external memory...\n");

        HANDLE sharedHandle;
        pDevice->CreateSharedHandle(pBuffer.Get(), nullptr, GENERIC_ALL, nullptr, &sharedHandle);

        cudaExternalMemoryHandleDesc extDesc = {};
        extDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        extDesc.flags = cudaExternalMemoryDedicated;
        extDesc.size = fileSize;
        extDesc.handle.win32.handle = sharedHandle;

        cudaImportExternalMemory(&extMemFile, &extDesc);
        CloseHandle(sharedHandle);

        cudaExternalMemoryBufferDesc bufDesc = {};
        bufDesc.size = fileSize;
        cudaExternalMemoryGetMappedBuffer(&d_fileData, extMemFile, &bufDesc);

        // --- 6. Generate Line Offsets (GPU - Simple Atomic Approach) ---
        printf("[Prep] Finding line offsets on GPU...\n");
        QueryPerformanceCounter(&start_offsets);

        // Calculate grid size for scanning
        int scan_blocks = 2048;  // Good parallelism
        int scan_threads = 256;

        // Estimate max lines (conservative: avg 10 bytes per line)
        size_t max_lines = fileSize / 10 + 1;

        // Allocate offsets array and counter
        ThrowIfCudaFailed(cudaMalloc(&d_lineOffsets, (max_lines + 1) * sizeof(size_t)),
            "Malloc line offsets");

        size_t* d_counter;
        ThrowIfCudaFailed(cudaMalloc(&d_counter, sizeof(size_t)), "Malloc counter");
        ThrowIfCudaFailed(cudaMemset(d_counter, 0, sizeof(size_t)), "Reset counter");

        // Set first offset to 0
        size_t zero = 0;
        ThrowIfCudaFailed(cudaMemcpy(d_lineOffsets, &zero, sizeof(size_t),
            cudaMemcpyHostToDevice), "Set first offset");

        // Build offsets array
        build_offsets_kernel_atomic << <scan_blocks, scan_threads >> > (
            (char*)d_fileData, fileSize, d_lineOffsets, d_counter
            );
        ThrowIfCudaFailed(cudaDeviceSynchronize(), "Build offsets kernel");

        // Get actual line count
        size_t num_lines;
        ThrowIfCudaFailed(cudaMemcpy(&num_lines, d_counter, sizeof(size_t),
            cudaMemcpyDeviceToHost), "Get line count");

        printf("      Found %llu lines\n", num_lines);

        // Set last offset to fileSize
        ThrowIfCudaFailed(cudaMemcpy(d_lineOffsets + num_lines + 1, &fileSize, sizeof(size_t),
            cudaMemcpyHostToDevice), "Set last offset");

        // Cleanup counter
        cudaFree(d_counter);

        QueryPerformanceCounter(&end_offsets);
        printf("      Line offset generation completed in %.2f ms\n", get_elapsed_ms(start_offsets, end_offsets));

        // --- 7. Allocate CUDA Working Memory ---
        printf("[CUDA] Allocating working memory...\n");

        int num_blocks = 256;
        int num_threads = num_blocks * BLOCK_SIZE;

        ThrowIfCudaFailed(cudaMalloc(&d_globalHash, HASH_SIZE * sizeof(StationStats)),
            "Malloc global hash");
        ThrowIfCudaFailed(cudaMemset(d_globalHash, 0, HASH_SIZE * sizeof(StationStats)),
            "Memset global hash");

        ThrowIfCudaFailed(cudaMalloc(&d_threadResults,
            (size_t)num_threads * LOCAL_HASH_SIZE * sizeof(StationStats)),
            "Malloc thread results");
        ThrowIfCudaFailed(cudaMalloc(&d_threadCounts, num_threads * sizeof(int)),
            "Malloc thread counts");

        // --- 8. Execute Kernels ---
        printf("[CUDA] Processing measurements...\n");
        QueryPerformanceCounter(&start_compute);

        ThrowIfCudaFailed(cudaMemset(d_threadResults, 0,
            (size_t)num_threads * LOCAL_HASH_SIZE * sizeof(StationStats)),
            "Reset thread results");
        ThrowIfCudaFailed(cudaMemset(d_threadCounts, 0, num_threads * sizeof(int)),
            "Reset thread counts");

        process_measurements_local << <num_blocks, BLOCK_SIZE >> > (
            (char*)d_fileData, d_lineOffsets, num_lines,
            d_threadResults, d_threadCounts
            );
        ThrowIfCudaFailed(cudaDeviceSynchronize(), "Process kernel");

        printf("[CUDA] Merging results...\n");
        int merge_blocks = (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        merge_results << <merge_blocks, BLOCK_SIZE >> > (
            d_threadResults, d_threadCounts, num_threads, d_globalHash
            );
        ThrowIfCudaFailed(cudaDeviceSynchronize(), "Merge kernel");

        QueryPerformanceCounter(&end_compute);
        printf("      Computation completed in %.2f ms\n", get_elapsed_ms(start_compute, end_compute));

        // --- 9. Copy Results Back ---
        printf("[Out] Reading back results...\n");
        QueryPerformanceCounter(&start_readback);

        StationStats* h_globalHash = (StationStats*)malloc(HASH_SIZE * sizeof(StationStats));
        ThrowIfCudaFailed(cudaMemcpy(h_globalHash, d_globalHash, HASH_SIZE * sizeof(StationStats),
            cudaMemcpyDeviceToHost), "Copy results to host");

        QueryPerformanceCounter(&end_readback);
        printf("      Readback completed in %.2f ms\n", get_elapsed_ms(start_readback, end_readback));

        // --- 10. Sort and Write Output ---
        printf("[Out] Writing output...\n");

        std::map<std::string, StationStats> sortedStats;
        for (int i = 0; i < HASH_SIZE; i++) {
            if (h_globalHash[i].count > 0 && h_globalHash[i].name[0] != '\0') {
                std::string key(h_globalHash[i].name);
                auto it = sortedStats.find(key);
                if (it != sortedStats.end()) {
                    // Merge duplicates (hash collisions)
                    if (h_globalHash[i].min < it->second.min) it->second.min = h_globalHash[i].min;
                    if (h_globalHash[i].max > it->second.max) it->second.max = h_globalHash[i].max;
                    it->second.sum += h_globalHash[i].sum;
                    it->second.count += h_globalHash[i].count;
                }
                else {
                    sortedStats[key] = h_globalHash[i];
                }
            }
        }

        std::ofstream outfile(outputFileName);
        outfile << "{";
        bool first = true;
        for (const auto& kv : sortedStats) {
            if (!first) outfile << ", ";
            const auto& s = kv.second;
            double minVal = s.min / 10.0;
            double avgVal = (s.sum / 10.0) / s.count;
            double maxVal = s.max / 10.0;
            outfile << kv.first << "="
                << std::fixed << std::setprecision(1)
                << minVal << "/" << avgVal << "/" << maxVal;
            first = false;
        }
        outfile << "}\n";
        outfile.close();

        printf("      Wrote %zu stations to %s\n", sortedStats.size(), outputFileName);

        // --- Cleanup ---
        free(h_globalHash);
        cudaFree(d_lineOffsets);
        cudaFree(d_threadResults);
        cudaFree(d_threadCounts);
        cudaFree(d_globalHash);

        QueryPerformanceCounter(&end_total);

        // Print timing summary
        printf("\n========== TIMING SUMMARY ==========\n");
        printf("DirectStorage Load:  %8.2f ms\n", get_elapsed_ms(start_io, end_io));
        printf("Line Offset Gen:     %8.2f ms\n", get_elapsed_ms(start_offsets, end_offsets));
        printf("Computation:         %8.2f ms\n", get_elapsed_ms(start_compute, end_compute));
        printf("Readback:            %8.2f ms\n", get_elapsed_ms(start_readback, end_readback));
        printf("------------------------------------\n");
        printf("Total Runtime:       %8.2f ms (%.3f sec)\n",
            get_elapsed_ms(start_total, end_total),
            get_elapsed_ms(start_total, end_total) / 1000.0);
        printf("====================================\n");

    }
    catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }

    // Final cleanup
    if (extMemFile) cudaDestroyExternalMemory(extMemFile);
    if (fenceEvent) CloseHandle(fenceEvent);

    return 0;
}
