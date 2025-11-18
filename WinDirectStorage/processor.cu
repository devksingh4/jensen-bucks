/*
 * Copyright (c) 2024, NVIDIA CORPORATION / Microsoft DirectStorage
 *
 * This sample loads a "very large" file by enqueuing it in chunks.
 * It reads the file size dynamically, creates one large GPU buffer,
 * and enqueues N requests to fill it. The CUDA kernel runs once at the end.
 *
 * NOTE: This method requires VRAM >= File Size.
 *
 * Compilation:
 * nvcc processor.cu -o ds_cuda_interop.exe -I"path\to\include" -L"path\to\lib\x64" -ld3d12 -ldxgi -ldstorage -lcudart
 */

#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <comdef.h> // For _com_error
#include <algorithm> // For min/max

 // Windows & D3D12
#include <Windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h> // For ComPtr

// DirectStorage
#include <dstorage.h>

// CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using Microsoft::WRL::ComPtr;
using std::min;
using std::max;

// --- Configuration ---
const char* FILE_NAME = "test_data.bin";
// File size is now read dynamically.
const uint64_t RESULT_SIZE_BYTES = sizeof(int64_t); // Kernel returns a 64-bit sum
const int THREADS_PER_BLOCK = 1024;

// Chunk size for enqueueing requests
const uint64_t CHUNK_SIZE = (16 * 1024 * 1024); // 16MB chunks

// --- Helper for HRESULT failures ---
inline void ThrowIfFailed(HRESULT hr, const char* msg) {
    if (FAILED(hr)) {
        _com_error err(hr);
        std::wcerr << L"FATAL ERROR: " << msg << L"\nHRESULT: 0x" << std::hex << hr
            << L" (" << err.ErrorMessage() << L")" << std::endl;
        throw std::runtime_error(msg);
    }
}

// --- Helper for CUDA API failures ---
inline void ThrowIfCudaFailed(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "FATAL ERROR: " << msg << "\nCUDA Error: "
            << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

// --- CUDA Kernel ---
__global__ void sum_kernel(const int32_t* g_data, int64_t* d_result, uint32_t numElements) {
    extern __shared__ int64_t s_data[];
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    // 1. Grid-Stride Loop (Per-Thread Reduction)
    int64_t thread_sum = 0;
    for (unsigned int i = tid; i < numElements; i += block_size) {
        thread_sum += g_data[i];
    }
    s_data[tid] = thread_sum;
    __syncthreads();

    // 2. Shared Memory Reduction
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // 3. Write Result
    if (tid == 0) {
        *d_result = s_data[0];
    }
}

int main() {
    printf("Starting DirectStorage (Large File Chunking) -> CUDA Interop Test...\n");
    fflush(stdout);

    // Resources
    ComPtr<ID3D12Device> pDevice;
    ComPtr<IDXGIFactory4> pDxgiFactory;
    ComPtr<ID3D12CommandQueue> pCommandQueue;
    ComPtr<ID3D12CommandAllocator> pCommandAllocator;
    ComPtr<ID3D12GraphicsCommandList> pCommandList;
    ComPtr<ID3D12Fence> pFence;
    ComPtr<ID3D12Resource> pBuffer, pResultBuffer, pReadbackBuffer;
    ComPtr<IDStorageFactory> pDsFactory;
    ComPtr<IDStorageFile> pDsFile;
    ComPtr<IDStorageQueue> pDsQueue;

    HANDLE fenceEvent;
    UINT64 fenceValue = 1;
    int cudaDeviceID = -1;

    cudaExternalMemory_t extMemFile = nullptr;
    cudaExternalMemory_t extMemResult = nullptr;
    void* d_fileData = nullptr;
    void* d_result = nullptr;

    // File size is now dynamic
    uint64_t actualFileSize = 0;
    uint32_t actualNumInts = 0;


    try {
        // 1. Init D3D12
        printf("1. Initializing D3D12 Device...\n");
        ThrowIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(&pDxgiFactory)), "CreateDXGIFactory2 failed");

        ComPtr<IDXGIAdapter1> pAdapter;
        DXGI_ADAPTER_DESC1 adapterDesc = {};
        for (UINT i = 0; pDxgiFactory->EnumAdapters1(i, &pAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            pAdapter->GetDesc1(&adapterDesc);
            if (adapterDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (SUCCEEDED(D3D12CreateDevice(pAdapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr))) break;
        }
        ThrowIfFailed(D3D12CreateDevice(pAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pDevice)), "D3D12CreateDevice failed");

        // 2. Init CUDA
        printf("2. Finding matching CUDA Device...\n");
        int numCudaDevices = 0;
        cudaGetDeviceCount(&numCudaDevices);
        for (int i = 0; i < numCudaDevices; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            if (memcmp(prop.luid, &adapterDesc.AdapterLuid, sizeof(adapterDesc.AdapterLuid)) == 0) {
                cudaDeviceID = i;
                break;
            }
        }
        if (cudaDeviceID == -1) throw std::runtime_error("No matching CUDA device found.");
        ThrowIfCudaFailed(cudaSetDevice(cudaDeviceID), "cudaSetDevice failed");

        // 3. Init DirectStorage & Get File Size
        printf("3. Initializing DirectStorage and reading file size...\n");
        ThrowIfFailed(DStorageGetFactory(IID_PPV_ARGS(&pDsFactory)), "DStorageGetFactory failed");

        std::wstring wFileName(FILE_NAME, FILE_NAME + strlen(FILE_NAME));
        ThrowIfFailed(pDsFactory->OpenFile(wFileName.c_str(), IID_PPV_ARGS(&pDsFile)), "OpenFile failed");

        BY_HANDLE_FILE_INFORMATION fileInfo = {};
        ThrowIfFailed(pDsFile->GetFileInformation(&fileInfo), "GetFileInformation failed");
        actualFileSize = (static_cast<uint64_t>(fileInfo.nFileSizeHigh) << 32) | fileInfo.nFileSizeLow;
        actualNumInts = (uint32_t)(actualFileSize / sizeof(int32_t));

        std::cout << "   File size: " << actualFileSize << " bytes ("
            << (actualFileSize / (1024.0 * 1024.0)) << " MB)" << std::endl;

        if (actualFileSize == 0) {
            throw std::runtime_error("File is empty.");
        }

        // 4. Create D3D12 Buffers
        printf("4. Creating D3D12 Buffers...\n");
        D3D12_COMMAND_QUEUE_DESC queueDesc = { D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0 };
        ThrowIfFailed(pDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&pCommandQueue)), "CreateCommandQueue failed");
        ThrowIfFailed(pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&pCommandAllocator)), "CreateCommandAllocator failed");
        ThrowIfFailed(pDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, pCommandAllocator.Get(), nullptr, IID_PPV_ARGS(&pCommandList)), "CreateCommandList failed");
        pCommandList->Close();
        ThrowIfFailed(pDevice->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&pFence)), "CreateFence failed");
        fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

        D3D12_HEAP_PROPERTIES heapProps = { D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1 };
        D3D12_RESOURCE_DESC resDesc = {};
        resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resDesc.Width = actualFileSize; // <--- DYNAMIC SIZE
        resDesc.Height = 1;
        resDesc.DepthOrArraySize = 1;
        resDesc.MipLevels = 1;
        resDesc.SampleDesc.Count = 1;
        resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        // GPU Buffer (File)
        ThrowIfFailed(pDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_SHARED, &resDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&pBuffer)), "Create Buffer failed");

        // GPU Buffer (Result)
        resDesc.Width = RESULT_SIZE_BYTES;
        ThrowIfFailed(pDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_SHARED, &resDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&pResultBuffer)), "Create Result failed");

        // CPU Buffer (Readback)
        heapProps.Type = D3D12_HEAP_TYPE_READBACK;
        resDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
        ThrowIfFailed(pDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&pReadbackBuffer)), "Create Readback failed");

        // 5. Create DS Queue
        const uint64_t numChunks = (actualFileSize + CHUNK_SIZE - 1) / CHUNK_SIZE;
        std::cout << "   Splitting into " << numChunks << " chunks" << std::endl;

        const uint32_t queueCapacity =
            min(DSTORAGE_MAX_QUEUE_CAPACITY, max(DSTORAGE_MIN_QUEUE_CAPACITY, (uint32_t)(numChunks * 2)));

        DSTORAGE_QUEUE_DESC dsQueueDesc{};
        dsQueueDesc.Capacity = queueCapacity;
        dsQueueDesc.Priority = DSTORAGE_PRIORITY_NORMAL;
        dsQueueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        dsQueueDesc.Device = pDevice.Get();
        ThrowIfFailed(pDsFactory->CreateQueue(&dsQueueDesc, IID_PPV_ARGS(&pDsQueue)), "CreateQueue failed");
        std::cout << "   Queue capacity set to " << queueCapacity << " requests" << std::endl;


        // 6. Enqueue Requests (Chunked)
        printf("6. Enqueuing %llu requests...\n", numChunks);
        for (uint64_t i = 0; i < numChunks; ++i)
        {
            DSTORAGE_REQUEST request = {};
            request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;

            // [FIX] Use the correct member as you pointed out
            request.Source.File.Source = pDsFile.Get();

            // Calculate offset and size for this chunk
            uint64_t offset = i * CHUNK_SIZE;
            uint32_t chunkSize = (uint32_t)min(CHUNK_SIZE, actualFileSize - offset);

            request.Source.File.Offset = offset;
            request.Source.File.Size = chunkSize;

            // Uncompressed transfer
            request.UncompressedSize = chunkSize;
            request.Destination.Buffer.Resource = pBuffer.Get();
            request.Destination.Buffer.Offset = offset;
            request.Destination.Buffer.Size = chunkSize;

            pDsQueue->EnqueueRequest(&request);
        }

        // 7. Wait for Completion
        printf("7. Submitting and waiting for DS completion...\n");
        pDsQueue->EnqueueSignal(pFence.Get(), fenceValue);
        pDsQueue->Submit();
        if (pFence->GetCompletedValue() < fenceValue) {
            pFence->SetEventOnCompletion(fenceValue, fenceEvent);
            WaitForSingleObject(fenceEvent, INFINITE);
        }
        fenceValue++;
        printf("   DS load complete.\n");

        // 8. Import to CUDA
        printf("8. Importing to CUDA...\n");
        cudaExternalMemoryHandleDesc extMemHandleDesc = {};
        extMemHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        extMemHandleDesc.flags = cudaExternalMemoryDedicated;
        HANDLE sharedHandle;

        // Import File Buffer
        ThrowIfFailed(pDevice->CreateSharedHandle(pBuffer.Get(), nullptr, GENERIC_ALL, nullptr, &sharedHandle), "SharedHandle failed");
        extMemHandleDesc.handle.win32.handle = sharedHandle;
        extMemHandleDesc.size = actualFileSize; // <--- DYNAMIC SIZE
        ThrowIfCudaFailed(cudaImportExternalMemory(&extMemFile, &extMemHandleDesc), "cudaImportExternalMemory failed");
        CloseHandle(sharedHandle);

        // Import Result Buffer
        ThrowIfFailed(pDevice->CreateSharedHandle(pResultBuffer.Get(), nullptr, GENERIC_ALL, nullptr, &sharedHandle), "SharedHandle failed");
        extMemHandleDesc.handle.win32.handle = sharedHandle;
        extMemHandleDesc.size = RESULT_SIZE_BYTES;
        ThrowIfCudaFailed(cudaImportExternalMemory(&extMemResult, &extMemHandleDesc), "cudaImportExternalMemory failed");
        CloseHandle(sharedHandle);

        // 9. Map Pointers
        printf("9. Mapping Pointers...\n");
        cudaExternalMemoryBufferDesc bufferDesc = {};
        bufferDesc.offset = 0;
        bufferDesc.flags = 0;
        bufferDesc.size = actualFileSize; // <--- DYNAMIC SIZE
        ThrowIfCudaFailed(cudaExternalMemoryGetMappedBuffer(&d_fileData, extMemFile, &bufferDesc), "Map File failed");

        bufferDesc.size = RESULT_SIZE_BYTES;
        ThrowIfCudaFailed(cudaExternalMemoryGetMappedBuffer(&d_result, extMemResult, &bufferDesc), "Map Result failed");

        // 10. Launch Kernel
        printf("10. Launching Kernel...\n");
        size_t sharedMemSize = THREADS_PER_BLOCK * sizeof(int64_t);
        sum_kernel << <1, THREADS_PER_BLOCK, sharedMemSize >> > ((int32_t*)d_fileData, (int64_t*)d_result, actualNumInts);
        ThrowIfCudaFailed(cudaDeviceSynchronize(), "Kernel execution failed");
        printf("   Kernel execution complete.\n");

        // 11. Unmap (Implicit, just don't free pointers)
        d_fileData = nullptr;
        d_result = nullptr;

        // 12. Copy Result
        printf("12. Copying result to Readback...\n");
        pCommandAllocator->Reset();
        pCommandList->Reset(pCommandAllocator.Get(), nullptr);

        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = pResultBuffer.Get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        pCommandList->ResourceBarrier(1, &barrier);

        pCommandList->CopyResource(pReadbackBuffer.Get(), pResultBuffer.Get());

        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        pCommandList->ResourceBarrier(1, &barrier);

        pCommandList->Close();
        ID3D12CommandList* ppCommandLists[] = { pCommandList.Get() };
        pCommandQueue->ExecuteCommandLists(1, ppCommandLists);

        pCommandQueue->Signal(pFence.Get(), fenceValue);
        if (pFence->GetCompletedValue() < fenceValue) {
            pFence->SetEventOnCompletion(fenceValue, fenceEvent);
            WaitForSingleObject(fenceEvent, INFINITE);
        }

        // 13. Read Result
        printf("13. Reading Result...\n");
        int64_t finalSum = 0;
        void* pCpuData;
        D3D12_RANGE readRange = { 0, RESULT_SIZE_BYTES };
        pReadbackBuffer->Map(0, &readRange, &pCpuData);
        finalSum = *(int64_t*)pCpuData;
        pReadbackBuffer->Unmap(0, nullptr);

        printf("\n----------------------------------\n");
        printf("           FINAL SUM: %lld\n", finalSum);
        printf("----------------------------------\n");
    }
    catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
    }

    // Cleanup
    printf("Cleaning up resources...\n");
    // Wait for all GPU work to finish
    if (pCommandQueue && pFence) {
        pCommandQueue->Signal(pFence.Get(), fenceValue);
        if (pFence->GetCompletedValue() < fenceValue) {
            pFence->SetEventOnCompletion(fenceValue, fenceEvent);
            WaitForSingleObject(fenceEvent, INFINITE);
        }
    }

    if (extMemFile) cudaDestroyExternalMemory(extMemFile);
    if (extMemResult) cudaDestroyExternalMemory(extMemResult);
    if (fenceEvent) CloseHandle(fenceEvent);

    return 0;
}