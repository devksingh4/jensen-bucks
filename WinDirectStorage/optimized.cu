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
// CONFIGURATION & STRUCTS
// =================================================================================

#define MAX_NAME_LEN 100
#define HASH_SIZE 16384
#define LOCAL_HASH_SIZE 32  // Reduced from 64
#define BLOCK_SIZE 256

const int NUM_STREAM_BUFFERS = 3;
const uint64_t STREAM_CHUNK_SIZE = 256 * 1024 * 1024; // 256MB chunks
const size_t OVERLAP_SIZE = 4096; // 4KB overlap for boundary detection

struct StationStats {
    char name[MAX_NAME_LEN];
    int min;
    int max;
    int64_t sum;
    int count;
    unsigned int hash;
    size_t name_len;
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
    
    // Per-chunk working memory
    size_t* d_lineOffsets;
    size_t* d_counter;
    size_t* d_lastNewline;
    StationStats* d_threadResults;
    int* d_threadCounts;
    
    size_t maxLinesCapacity;
    uint64_t actualDataSize;
    bool inUse;
};

// =================================================================================
// DEVICE HELPER FUNCTIONS
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
    while (i < line_len && line[i] != ';') {
        station[i] = line[i];
        i++;
    }
    *station_len = i;
    station[i] = '\0';

    i++;
    if (i >= line_len) return;

    int sign = 1;
    int temp_val = 0;

    if (line[i] == '-') {
        sign = -1;
        i++;
    }

    if (i + 1 < line_len && line[i + 1] == '.') {
        temp_val = (line[i] - '0') * 10 + (line[i + 2] - '0');
    }
    else if (i + 2 < line_len && line[i + 2] == '.') {
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
// CUDA KERNELS
// =================================================================================

__global__ void build_offsets_kernel_atomic(const char* data, size_t data_size,
    size_t* offsets, size_t* counter) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t i = idx; i < data_size; i += stride) {
        if (data[i] == '\n' && i + 1 < data_size) {
            size_t pos = atomicAdd((unsigned long long*)counter, 1ULL);
            offsets[pos + 1] = i + 1;
        }
    }
}

__global__ void find_last_newline(const char* data, size_t size, size_t* result) {
    // Search backwards from end for last newline
    for (size_t i = size - 1; i > 0 && i > size - OVERLAP_SIZE; i--) {
        if (data[i] == '\n') {
            *result = i + 1;
            return;
        }
    }
    *result = size;
}

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
                        atomicAdd((unsigned long long*)&global_hash[current_slot].sum,
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
// MAIN
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

    auto get_elapsed_ms = [&freq](LARGE_INTEGER start, LARGE_INTEGER end) -> double {
        return (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
    };

    LARGE_INTEGER start_total, end_total;
    LARGE_INTEGER start_stream, end_stream;

    QueryPerformanceCounter(&start_total);

    // D3D12 & DS Objects
    ComPtr<ID3D12Device> pDevice;
    ComPtr<IDXGIFactory4> pDxgiFactory;
    ComPtr<IDStorageFactory> pDsFactory;
    ComPtr<IDStorageFile> pDsFile;
    ComPtr<IDStorageQueue> pDsQueue;

    // CUDA Objects
    StationStats* d_globalHash = nullptr;
    StreamBuffer streamBuffers[NUM_STREAM_BUFFERS];

    try {
        // === 1. Init D3D12 & CUDA ===
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

        // === 2. Init DirectStorage & Get File Info ===
        printf("[IO] Opening file: %s\n", inputFileName);

        DStorageGetFactory(IID_PPV_ARGS(&pDsFactory));

        std::wstring wFileName(inputFileName, inputFileName + strlen(inputFileName));
        ThrowIfFailed(pDsFactory->OpenFile(wFileName.c_str(), IID_PPV_ARGS(&pDsFile)), "OpenFile");

        BY_HANDLE_FILE_INFORMATION fileInfo = {};
        pDsFile->GetFileInformation(&fileInfo);
        uint64_t fileSize = (static_cast<uint64_t>(fileInfo.nFileSizeHigh) << 32) | fileInfo.nFileSizeLow;

        printf("     Size: %llu bytes (%.2f GB)\n", fileSize, fileSize / (1024.0 * 1024.0 * 1024.0));

        // === 3. Create DirectStorage Queue ===
        DSTORAGE_QUEUE_DESC dsQueueDesc = {};
        dsQueueDesc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
        dsQueueDesc.Priority = DSTORAGE_PRIORITY_HIGH;
        dsQueueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        dsQueueDesc.Device = pDevice.Get();

        pDsFactory->CreateQueue(&dsQueueDesc, IID_PPV_ARGS(&pDsQueue));

        // Set staging buffer size
        pDsFactory->SetStagingBufferSize(512 * 1024 * 1024);

        // === 4. Initialize Stream Buffers ===
        printf("[Init] Creating %d stream buffers...\n", NUM_STREAM_BUFFERS);

        int num_blocks = 256;
        int num_threads = num_blocks * BLOCK_SIZE;
        size_t max_lines_per_chunk = (STREAM_CHUNK_SIZE / 10) + 1000;

        for (int i = 0; i < NUM_STREAM_BUFFERS; i++) {
            auto& buf = streamBuffers[i];

            // Create D3D12 buffer
            D3D12_HEAP_PROPERTIES heapProps = {
                D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                D3D12_MEMORY_POOL_UNKNOWN, 1, 1
            };

            D3D12_RESOURCE_DESC resDesc = {};
            resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            resDesc.Width = STREAM_CHUNK_SIZE + OVERLAP_SIZE;
            resDesc.Height = 1;
            resDesc.DepthOrArraySize = 1;
            resDesc.MipLevels = 1;
            resDesc.SampleDesc.Count = 1;
            resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

            ThrowIfFailed(pDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_SHARED,
                &resDesc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                IID_PPV_ARGS(&buf.d3d12Buffer)), "Create stream buffer");

            // Create fence
            pDevice->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&buf.d3dFence));
            buf.fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
            buf.fenceValue = 0;

            // Import to CUDA
            HANDLE sharedHandle;
            pDevice->CreateSharedHandle(buf.d3d12Buffer.Get(), nullptr,
                GENERIC_ALL, nullptr, &sharedHandle);

            cudaExternalMemoryHandleDesc extDesc = {};
            extDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
            extDesc.flags = cudaExternalMemoryDedicated;
            extDesc.size = STREAM_CHUNK_SIZE + OVERLAP_SIZE;
            extDesc.handle.win32.handle = sharedHandle;

            ThrowIfCudaFailed(cudaImportExternalMemory(&buf.cudaExtMem, &extDesc),
                "Import external memory");
            CloseHandle(sharedHandle);

            cudaExternalMemoryBufferDesc bufDesc = {};
            bufDesc.size = STREAM_CHUNK_SIZE + OVERLAP_SIZE;
            ThrowIfCudaFailed(cudaExternalMemoryGetMappedBuffer(&buf.cudaPtr,
                buf.cudaExtMem, &bufDesc), "Map buffer to CUDA");

            // Create CUDA stream
            ThrowIfCudaFailed(cudaStreamCreate(&buf.cudaStream), "Create CUDA stream");
            ThrowIfCudaFailed(cudaEventCreate(&buf.processingComplete), "Create event");

            // Allocate working memory
            buf.maxLinesCapacity = max_lines_per_chunk;

            ThrowIfCudaFailed(cudaMalloc(&buf.d_lineOffsets,
                (buf.maxLinesCapacity + 2) * sizeof(size_t)), "Malloc line offsets");
            ThrowIfCudaFailed(cudaMalloc(&buf.d_counter, sizeof(size_t)), "Malloc counter");
            ThrowIfCudaFailed(cudaMalloc(&buf.d_lastNewline, sizeof(size_t)), "Malloc last newline");

            ThrowIfCudaFailed(cudaMalloc(&buf.d_threadResults,
                (size_t)num_threads * LOCAL_HASH_SIZE * sizeof(StationStats)),
                "Malloc thread results");
            ThrowIfCudaFailed(cudaMalloc(&buf.d_threadCounts,
                num_threads * sizeof(int)), "Malloc thread counts");

            buf.inUse = false;
        }

        // === 5. Allocate Global Hash ===
        ThrowIfCudaFailed(cudaMalloc(&d_globalHash, HASH_SIZE * sizeof(StationStats)),
            "Malloc global hash");
        ThrowIfCudaFailed(cudaMemset(d_globalHash, 0, HASH_SIZE * sizeof(StationStats)),
            "Memset global hash");

        // === 6. Process File in Chunks ===
        uint64_t numChunks = (fileSize + STREAM_CHUNK_SIZE - 1) / STREAM_CHUNK_SIZE;
        printf("[Stream] Processing file in %llu chunks of %llu MB...\n",
            numChunks, STREAM_CHUNK_SIZE / (1024 * 1024));

        QueryPerformanceCounter(&start_stream);

        int currentBuffer = 0;
        UINT64 nextFenceValue = 1;
        uint64_t currentFileOffset = 0;

        while (currentFileOffset < fileSize) {
            // === Find Available Buffer ===
            StreamBuffer* buf = nullptr;
            int attempts = 0;

            while (buf == nullptr) {
                StreamBuffer& candidate = streamBuffers[currentBuffer];

                if (!candidate.inUse) {
                    buf = &candidate;
                }
                else {
                    if (candidate.d3dFence->GetCompletedValue() >= candidate.fenceValue) {
                        cudaError_t err = cudaEventQuery(candidate.processingComplete);
                        if (err == cudaSuccess) {
                            candidate.inUse = false;
                            buf = &candidate;
                        }
                        else if (err != cudaErrorNotReady) {
                            ThrowIfCudaFailed(err, "Event query");
                        }
                    }
                }

                if (buf == nullptr) {
                    currentBuffer = (currentBuffer + 1) % NUM_STREAM_BUFFERS;
                    attempts++;
                    if (attempts >= NUM_STREAM_BUFFERS) {
                        StreamBuffer& oldest = streamBuffers[currentBuffer];
                        if (oldest.d3dFence->GetCompletedValue() < oldest.fenceValue) {
                            oldest.d3dFence->SetEventOnCompletion(oldest.fenceValue, oldest.fenceEvent);
                            WaitForSingleObject(oldest.fenceEvent, INFINITE);
                        }
                        ThrowIfCudaFailed(cudaEventSynchronize(oldest.processingComplete),
                            "Wait for processing");
                        oldest.inUse = false;
                        buf = &oldest;
                    }
                }
            }

            // === Load Chunk ===
            uint64_t readSize = min(STREAM_CHUNK_SIZE + OVERLAP_SIZE, fileSize - currentFileOffset);

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

            // === Wait for Load ===
            if (buf->d3dFence->GetCompletedValue() < buf->fenceValue) {
                buf->d3dFence->SetEventOnCompletion(buf->fenceValue, buf->fenceEvent);
                WaitForSingleObject(buf->fenceEvent, INFINITE);
            }

            // === Find Last Complete Line ===
            size_t processableSize = readSize;
            
            if (currentFileOffset + readSize < fileSize) {
                find_last_newline<<<1, 1, 0, buf->cudaStream>>>(
                    (char*)buf->cudaPtr, readSize, buf->d_lastNewline
                );
                ThrowIfCudaFailed(cudaMemcpyAsync(&processableSize, buf->d_lastNewline,
                    sizeof(size_t), cudaMemcpyDeviceToHost, buf->cudaStream),
                    "Get processable size");
                ThrowIfCudaFailed(cudaStreamSynchronize(buf->cudaStream),
                    "Sync for processable size");
            }

            buf->actualDataSize = processableSize;

            // === Build Line Offsets ===
            ThrowIfCudaFailed(cudaMemsetAsync(buf->d_counter, 0, sizeof(size_t), buf->cudaStream),
                "Reset counter");

            size_t zero = 0;
            ThrowIfCudaFailed(cudaMemcpyAsync(buf->d_lineOffsets, &zero, sizeof(size_t),
                cudaMemcpyHostToDevice, buf->cudaStream), "Set first offset");

            int scan_blocks = min(2048, (int)((processableSize + BLOCK_SIZE - 1) / BLOCK_SIZE));
            build_offsets_kernel_atomic<<<scan_blocks, BLOCK_SIZE, 0, buf->cudaStream>>>(
                (char*)buf->cudaPtr, processableSize, buf->d_lineOffsets, buf->d_counter
            );

            // Get line count
            size_t num_lines;
            ThrowIfCudaFailed(cudaMemcpyAsync(&num_lines, buf->d_counter, sizeof(size_t),
                cudaMemcpyDeviceToHost, buf->cudaStream), "Get line count");
            ThrowIfCudaFailed(cudaStreamSynchronize(buf->cudaStream), "Wait for line count");

            // Set last offset
            ThrowIfCudaFailed(cudaMemcpyAsync(buf->d_lineOffsets + num_lines + 1,
                &processableSize, sizeof(size_t), cudaMemcpyHostToDevice, buf->cudaStream),
                "Set last offset");

            // === Process Lines ===
            ThrowIfCudaFailed(cudaMemsetAsync(buf->d_threadResults, 0,
                (size_t)num_threads * LOCAL_HASH_SIZE * sizeof(StationStats),
                buf->cudaStream), "Reset thread results");
            ThrowIfCudaFailed(cudaMemsetAsync(buf->d_threadCounts, 0,
                num_threads * sizeof(int), buf->cudaStream), "Reset thread counts");

            process_measurements_local<<<num_blocks, BLOCK_SIZE, 0, buf->cudaStream>>>(
                (char*)buf->cudaPtr, buf->d_lineOffsets, num_lines,
                buf->d_threadResults, buf->d_threadCounts
            );

            // === Merge Results ===
            int merge_blocks = (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
            merge_results<<<merge_blocks, BLOCK_SIZE, 0, buf->cudaStream>>>(
                buf->d_threadResults, buf->d_threadCounts, num_threads, d_globalHash
            );

            // Record completion
            ThrowIfCudaFailed(cudaEventRecord(buf->processingComplete, buf->cudaStream),
                "Record event");

            // Advance file offset
            currentFileOffset += processableSize;

            // Move to next buffer
            currentBuffer = (currentBuffer + 1) % NUM_STREAM_BUFFERS;

            // Progress update
            static int lastPercent = -1;
            int percent = (int)(100.0 * currentFileOffset / fileSize);
            if (percent != lastPercent && percent % 10 == 0) {
                printf("     Progress: %d%%\n", percent);
                lastPercent = percent;
            }
        }

        // === Wait for All Processing ===
        printf("[Stream] Waiting for all processing to complete...\n");
        for (int i = 0; i < NUM_STREAM_BUFFERS; i++) {
            if (streamBuffers[i].inUse) {
                ThrowIfCudaFailed(cudaEventSynchronize(streamBuffers[i].processingComplete),
                    "Final sync");
            }
        }

        QueryPerformanceCounter(&end_stream);
        printf("     Total streaming completed in %.2f ms\n", get_elapsed_ms(start_stream, end_stream));

        // === 7. Copy Results Back ===
        printf("[Out] Reading back results...\n");

        StationStats* h_globalHash = (StationStats*)malloc(HASH_SIZE * sizeof(StationStats));
        ThrowIfCudaFailed(cudaMemcpy(h_globalHash, d_globalHash, HASH_SIZE * sizeof(StationStats),
            cudaMemcpyDeviceToHost), "Copy results to host");

        // === 8. Sort and Write Output ===
        printf("[Out] Writing output...\n");

        std::map<std::string, StationStats> sortedStats;
        for (int i = 0; i < HASH_SIZE; i++) {
            if (h_globalHash[i].count > 0 && h_globalHash[i].name[0] != '\0') {
                std::string key(h_globalHash[i].name);
                auto it = sortedStats.find(key);
                if (it != sortedStats.end()) {
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

        // === Cleanup ===
        free(h_globalHash);
        
        for (int i = 0; i < NUM_STREAM_BUFFERS; i++) {
            auto& buf = streamBuffers[i];
            cudaFree(buf.d_lineOffsets);
            cudaFree(buf.d_counter);
            cudaFree(buf.d_lastNewline);
            cudaFree(buf.d_threadResults);
            cudaFree(buf.d_threadCounts);
            cudaEventDestroy(buf.processingComplete);
            cudaStreamDestroy(buf.cudaStream);
            cudaDestroyExternalMemory(buf.cudaExtMem);
            CloseHandle(buf.fenceEvent);
        }
        
        cudaFree(d_globalHash);

        QueryPerformanceCounter(&end_total);

        // Print timing summary
        printf("\n========== TIMING SUMMARY ==========\n");
        printf("Streaming & Processing:  %8.2f ms\n", get_elapsed_ms(start_stream, end_stream));
        printf("------------------------------------\n");
        printf("Total Runtime:           %8.2f ms (%.3f sec)\n",
            get_elapsed_ms(start_total, end_total),
            get_elapsed_ms(start_total, end_total) / 1000.0);
        printf("====================================\n");

    }
    catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}