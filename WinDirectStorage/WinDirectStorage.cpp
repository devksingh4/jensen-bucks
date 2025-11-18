//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//

#include <dstorage.h>
#include <dxgi1_4.h>
#include <winrt/base.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

using winrt::check_hresult;
using winrt::com_ptr;

struct handle_closer
{
    void operator()(HANDLE h) noexcept
    {
        assert(h != INVALID_HANDLE_VALUE);
        if (h)
        {
            CloseHandle(h);
        }
    }
};
using ScopedHandle = std::unique_ptr<void, handle_closer>;

void ShowHelpText()
{
    std::cout << "Reads the contents of a file and writes them out to a buffer on the GPU using DirectStorage."
              << std::endl;
    std::cout << "Supports both uncompressed and GDeflate compressed files with GPU decompression." << std::endl
              << std::endl;
    std::cout << "USAGE: HelloDirectStorage [path] [options]" << std::endl;
    std::cout << "  [path]            - Path to file to load" << std::endl;
    std::cout << "  -compressed       - File is GDeflate compressed (use GPU decompression)" << std::endl;
    std::cout << "  -uncompressed_size [size] - Uncompressed size in bytes (required with -compressed)" << std::endl
              << std::endl;
    std::cout << "Example (uncompressed): HelloDirectStorage data.bin" << std::endl;
    std::cout << "Example (compressed):   HelloDirectStorage data.bin.gdeflate -compressed -uncompressed_size 629145600"
              << std::endl
              << std::endl;
}

// The following example reads from a specified data file and writes the contents
// to a D3D12 buffer resource, with optional GPU decompression.
int wmain(int argc, wchar_t* argv[])
{
    if (argc < 2)
    {
        ShowHelpText();
        return -1;
    }

    bool useCompression = false;
    uint64_t uncompressedSize = 0;

    // Parse command line arguments
    const wchar_t* fileToLoad = argv[1];
    for (int i = 2; i < argc; i++)
    {
        std::wstring arg = argv[i];
        if (arg == L"-compressed")
        {
            useCompression = true;
        }
        else if (arg == L"-uncompressed_size" && i + 1 < argc)
        {
            uncompressedSize = std::stoull(argv[i + 1]);
            i++;
        }
    }

    if (useCompression && uncompressedSize == 0)
    {
        std::cout << "ERROR: -uncompressed_size must be specified when using -compressed" << std::endl;
        ShowHelpText();
        return -1;
    }

    com_ptr<ID3D12Device> device;
    check_hresult(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device)));

    com_ptr<IDStorageFactory> factory;
    check_hresult(DStorageGetFactory(IID_PPV_ARGS(factory.put())));

    // CRITICAL OPTIMIZATION #1: Increase staging buffer size
    // For GPU decompression, larger staging buffers are essential
    // Recommendation: 128MB minimum, 256MB for high-end GPUs
    const uint32_t STAGING_BUFFER_SIZE = 256 * 1024 * 1024; // 256 MB for GPU decompression
    factory->SetStagingBufferSize(STAGING_BUFFER_SIZE);
    std::cout << "Staging buffer size set to " << (STAGING_BUFFER_SIZE / (1024.0 * 1024.0)) << " MB" << std::endl;

    com_ptr<IDStorageFile> file;
    HRESULT hr = factory->OpenFile(fileToLoad, IID_PPV_ARGS(file.put()));
    if (FAILED(hr))
    {
        std::wcout << L"The file '" << fileToLoad << L"' could not be opened. HRESULT=0x" << std::hex << hr
                   << std::endl;
        ShowHelpText();
        return -1;
    }

    BY_HANDLE_FILE_INFORMATION info{};
    check_hresult(file->GetFileInformation(&info));

    // Combine high and low parts to get the full 64-bit file size (supports files > 4GB)
    uint64_t compressedFileSize = (static_cast<uint64_t>(info.nFileSizeHigh) << 32) | info.nFileSizeLow;

    std::cout << "File size: " << compressedFileSize << " bytes (" << (compressedFileSize / (1024.0 * 1024.0))
              << " MB, " << (compressedFileSize / (1024.0 * 1024.0 * 1024.0)) << " GB)" << std::endl;

    // Determine the size of the GPU buffer we need
    uint64_t bufferSize = useCompression ? uncompressedSize : compressedFileSize;

    if (useCompression)
    {
        std::cout << "Uncompressed size: " << uncompressedSize << " bytes (" << (uncompressedSize / (1024.0 * 1024.0))
                  << " MB)" << std::endl;
        double compressionRatio = (double)uncompressedSize / (double)compressedFileSize;
        std::cout << "Compression ratio: " << compressionRatio << ":1" << std::endl;
        std::cout << "GPU decompression will be used!" << std::endl;
    }

    // OPTIMIZATION #2: Use optimal chunk size for compressed data
    // For GPU decompression: keep compressed chunks above 64KB for efficiency
    // GDeflate works in 64KB tiles, so align with that
    const uint64_t CHUNK_SIZE =
        useCompression ? (4 * 1024 * 1024) : (16 * 1024 * 1024); // 4MB for compressed, 16MB for uncompressed
    const uint64_t numChunks = (compressedFileSize + CHUNK_SIZE - 1) / CHUNK_SIZE;

    std::cout << "Splitting into " << numChunks << " chunks" << std::endl;

    // OPTIMIZATION #3: Set queue capacity appropriately
    const uint32_t queueCapacity =
        min(DSTORAGE_MAX_QUEUE_CAPACITY, max(DSTORAGE_MIN_QUEUE_CAPACITY, (uint32_t)(numChunks * 2)));

    // Create a DirectStorage queue which will be used to load data into a
    // buffer on the GPU.
    DSTORAGE_QUEUE_DESC queueDesc{};
    queueDesc.Capacity = queueCapacity;
    queueDesc.Priority = DSTORAGE_PRIORITY_NORMAL;
    queueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    queueDesc.Device = device.get();

    com_ptr<IDStorageQueue> queue;
    check_hresult(factory->CreateQueue(&queueDesc, IID_PPV_ARGS(queue.put())));

    std::cout << "Queue capacity set to " << queueCapacity << " requests" << std::endl;

    // Create the ID3D12Resource buffer which will be populated with the decompressed file's contents
    D3D12_HEAP_PROPERTIES bufferHeapProps = {};
    bufferHeapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = bufferSize;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufferDesc.SampleDesc.Count = 1;

    com_ptr<ID3D12Resource> bufferResource;
    check_hresult(device->CreateCommittedResource(
        &bufferHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(bufferResource.put())));

    // Enqueue requests to read and optionally decompress the file
    uint64_t currentUncompressedOffset = 0;

    for (uint64_t i = 0; i < numChunks; ++i)
    {
        DSTORAGE_REQUEST request = {};
        request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
        request.Source.File.Source = file.get();

        // Calculate offset and size for this chunk (in compressed file)
        uint64_t compressedOffset = i * CHUNK_SIZE;
        uint32_t compressedChunkSize = (uint32_t)min(CHUNK_SIZE, compressedFileSize - compressedOffset);

        request.Source.File.Offset = compressedOffset;
        request.Source.File.Size = compressedChunkSize;

        if (useCompression)
        {
            // Enable GPU decompression
            request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_GDEFLATE;

            // Calculate uncompressed chunk size (estimate based on compression ratio)
            double estimatedRatio = (double)uncompressedSize / (double)compressedFileSize;
            uint64_t uncompressedChunkSize = (uint64_t)(compressedChunkSize * estimatedRatio);

            // Make sure we don't exceed the buffer
            if (currentUncompressedOffset + uncompressedChunkSize > uncompressedSize)
            {
                uncompressedChunkSize = uncompressedSize - currentUncompressedOffset;
            }

            request.UncompressedSize = uncompressedChunkSize;
            request.Destination.Buffer.Resource = bufferResource.get();
            request.Destination.Buffer.Offset = currentUncompressedOffset;
            request.Destination.Buffer.Size = uncompressedChunkSize;

            currentUncompressedOffset += uncompressedChunkSize;
        }
        else
        {
            // Uncompressed transfer
            request.UncompressedSize = compressedChunkSize;
            request.Destination.Buffer.Resource = bufferResource.get();
            request.Destination.Buffer.Offset = compressedOffset;
            request.Destination.Buffer.Size = compressedChunkSize;
        }

        queue->EnqueueRequest(&request);
    }

    // Configure a fence to be signaled when all requests are completed
    com_ptr<ID3D12Fence> fence;
    check_hresult(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.put())));

    ScopedHandle fenceEvent(CreateEvent(nullptr, FALSE, FALSE, nullptr));
    constexpr uint64_t fenceValue = 1;
    check_hresult(fence->SetEventOnCompletion(fenceValue, fenceEvent.get()));
    queue->EnqueueSignal(fence.get(), fenceValue);

    // Start timing before submission
    auto startTime = std::chrono::high_resolution_clock::now();

    // Tell DirectStorage to start executing all queued items.
    queue->Submit();

    // Wait for the submitted work to complete
    std::cout << "Waiting for the DirectStorage request to complete..." << std::endl;
    WaitForSingleObject(fenceEvent.get(), INFINITE);

    // End timing after completion
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

    // Check the status array for errors.
    DSTORAGE_ERROR_RECORD errorRecord{};
    queue->RetrieveErrorRecord(&errorRecord);
    if (FAILED(errorRecord.FirstFailure.HResult))
    {
        std::cout << "The DirectStorage request failed! HRESULT=0x" << std::hex << errorRecord.FirstFailure.HResult
                  << std::endl;
        std::cout << "Number of failures: " << std::dec << errorRecord.FailureCount << std::endl;
    }
    else
    {
        std::cout << "The DirectStorage request completed successfully!" << std::endl;
        std::cout << "\n=== TIMING RESULTS ===" << std::endl;
        std::cout << "Time to transfer/decompress: " << duration.count() << " microseconds" << std::endl;
        std::cout << "Time to transfer/decompress: " << (duration.count() / 1000.0) << " milliseconds" << std::endl;
        std::cout << "Time to transfer/decompress: " << (duration.count() / 1000000.0) << " seconds" << std::endl;

        // Calculate effective throughput (based on uncompressed data)
        double seconds = duration.count() / 1000000.0;
        double effectiveDataSize = useCompression ? uncompressedSize : compressedFileSize;
        double mbPerSecond = (effectiveDataSize / (1024.0 * 1024.0)) / seconds;
        double gbPerSecond = mbPerSecond / 1024.0;

        std::cout << "\n=== THROUGHPUT ===" << std::endl;
        std::cout << "Effective throughput (uncompressed): " << mbPerSecond << " MB/s" << std::endl;
        std::cout << "Effective throughput (uncompressed): " << gbPerSecond << " GB/s" << std::endl;

        if (useCompression)
        {
            double diskReadMBps = (compressedFileSize / (1024.0 * 1024.0)) / seconds;
            double diskReadGBps = diskReadMBps / 1024.0;
            std::cout << "Disk read throughput (compressed): " << diskReadMBps << " MB/s" << std::endl;
            std::cout << "Disk read throughput (compressed): " << diskReadGBps << " GB/s" << std::endl;
            std::cout << "\n*** BANDWIDTH AMPLIFICATION: " << (mbPerSecond / diskReadMBps) << "x ***" << std::endl;
        }

        std::cout << "\nNumber of chunks processed: " << numChunks << std::endl;
    }

    return 0;
}