# Baselines

These are some implementations that we found which were our baseline metrics to optimize from.

- [gpu_cudf](baselines/gpu_cudf): a naive implementation in the cuDF GPU dataframe library
- [gpu_nvidia](baselines/gpu_nvidia): a libcudf C++ implementation by NVIDIA of the 1brc
    - See [this GitHub repository](https://github.com/rapidsai/cudf/tree/84b64896a463486a56e5c86a1551eda5e15fdc61/cpp/examples/billion_rows) and [this NVIDIA blog post](https://developer.nvidia.com/blog/processing-one-billion-rows-of-data-with-rapids-cudf-pandas-accelerator-mode/)
- [gpu_ref](baselines/gpu_ref): The fastest reported time we found on GPU (17 seconds on a V100)
    - See [this blog post](https://tspeterkim.github.io/posts/cuda-1brc) by the author.
    - Note: this implementation performs a single-threaded iteration of the entire file to parse city names, which is not included in the final runtime. Therefore, the reported runtime is not an apples-to-apples comparison.