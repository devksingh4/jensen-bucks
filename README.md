# 1 Billion Row Challenge with DirectStorage

**Authors: [Amol Shah](https://www.linkedin.com/in/amolshahh), [Dev Singh](https://devksingh.com), [Yanni Zhuang](https://yannizhuang.com)**

A high-performance implementation of the 1 Billion Row Challenge using the Microsoft DirectStorage API for accelerated data movement.

We process the full 1 billion rows in **5.8 seconds** on an ASUS ROG Zephyrus G15 (2022) gaming laptop with an NVIDIA 3070Ti GPU and an AMD Ryzen 9 6900HS CPU.

## What is the 1 Billion Row Challenge?

The 1BRC, or One Billion Row Challenge, is a programming competition that challenges developers to process a large file as quickly as possible. The goal is to read a 14 GB text file containing one billion rows of "station name;temperature" data and calculate the minimum, average, and maximum temperature for each unique station. Originally in Java, the challenge has expanded to many languages and compute architectures. See [gunnarmorling/1brc](https://github.com/gunnarmorling/1brc) for the original challenge.

## What is DirectStorage?

DirectStorage is a storage API from Microsoft that enables developers to directly load data from the storage device to the GPU, bypassing the CPU. This eliminates CPU bottlenecks in the I/O pipeline. Originally designed for Xbox to reduce load times, DirectStorage excels at high-throughput data processing tasks.

## Motivation

GPU-accelerated processing would seem like a natural first step for tackling the 1BRC; after all, GPUs excel at parallel workloads. However, traditional GPU approaches hit a fundamental bottleneck: moving data to the GPU is *really* slow. 

The conventional pipeline looks like this:
1. CPU reads file from disk (~3-7 GB/s on NVMe)
2. CPU copies data to GPU over PCIe (~12-16 GB/s on PCIe 3.0, ~25-32 GB/s on PCIe 4.0)

For a 14 GB file, this PCIe transfer alone takes 1-2 seconds—before any processing begins. Many GPU implementations struggle because:
- The entire 14 GB file may not fit in VRAM (especially on consumer GPUs)
- CPU-to-GPU transfers become the bottleneck, not computation
- The GPU sits idle while waiting for data

DirectStorage eliminates this bottleneck: by loading data directly from NVMe into GPU, we bypass the CPU entirely. This API is part of a broader class of Direct Memory Access (DMA) APIs.

## Performance Comparison

While our implementation doesn't beat the fastest CPU solutions (which achieve sub-2 second times on high-end server hardware), DirectStorage-based GPU processing offers compelling advantages:

### Competitive on Consumer Hardware
- Our 5.8s time on a mobile RTX 3070 Ti is competitive with many CPU implementations
- Consumer laptops can now handle billion-row datasets without specialized hardware
- Democratizes large-scale data processing

### Scalability Headroom
- The official 1BRC tests use servers with 32+ cores, 128GB RAM, and two NVMe SSDs
- Our approach uses a single mobile GPU with 8GB VRAM
- On equivalent server-class hardware (RTX 6000 Ada, A100, etc.) with faster NVMe arrays, we'd likely see dramatic improvements

## Building and Running

### Prerequisites

#### Operating System
Ensure that your system:
1. Has Windows 11 
2. Has an NVMe SSD 
3. Reports the machine as DirectStorage-compatible

To determine DirectStorage-compatibility, run `fsutil bypassIo state C:\` in an **Administrator** command prompt (change the drive letter as needed). You should see `BypassIo on "C:\" is currently supported`. 

#### Visual Studio
Download and install **Visual Studio 2022** (Community, Professional, or Enterprise) from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/downloads/)

During installation, select the following workloads:
- **Desktop development with C++**

Ensure these individual components are included (usually selected by default):
- MSVC v142 or v143 (C++ x64/x86 build tools)
- Windows 10 or 11 SDK
- C++ CMake tools for Windows

#### NVIDIA CUDA Toolkit
Download and install **CUDA Toolkit 13.0 or later** from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

Requirements:
- CUDA version >= 13.0
- Compatible NVIDIA GPU with compute capability 3.5 or higher
- Ensure CUDA_PATH environment variable is set (installer usually does this automatically)

#### Verify Installation
Open a Command Prompt or PowerShell and verify:
```powershell
cmake --version
nvcc --version
cl.exe
```

If these tools are not found, you may need to enter a Developer Command Prompt or Developer PowerShell, which was provided by Visual Studio.

### Building the Code
1. Download DirectStorage SDK from [here](https://www.nuget.org/api/v2/package/Microsoft.Direct3D.DirectStorage/1.3.0) and replace `.nupkg` extension with `.zip`
2. Extract contents to a folder `directstorage` in the current folder
3. Run `cmake -B build -S .` to configure CMake
4. Run `cmake --build build --config Release`
5. In `build\Release\`, there will be the executable and associated DLLs needed to run the application