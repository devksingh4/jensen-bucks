#include <fstream>
#include <vector>
#include <iostream>

// 256M integers * 4 bytes/int = 1,024 MB = 1 GB
const uint32_t NUM_INTS = 1024 * 1024 * 1024;
const int32_t VALUE = 2;
const char* FILE_NAME = "test_data.bin";

int main() {
    std::cout << "Creating large test file (1 GB). This will take a moment..." << std::endl;

    // We can't allocate a 1GB vector. We'll write in chunks.
    const size_t CHUNK_SIZE_INTS = 1024 * 1024; // 1M ints (4MB) per chunk
    std::vector<int32_t> chunk(CHUNK_SIZE_INTS, VALUE);

    std::ofstream outfile(FILE_NAME, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Could not create file " << FILE_NAME << std::endl;
        return 1;
    }

    uint32_t intsWritten = 0;
    while (intsWritten < NUM_INTS) {
        outfile.write(reinterpret_cast<const char*>(chunk.data()), chunk.size() * sizeof(int32_t));
        intsWritten += CHUNK_SIZE_INTS;
    }

    outfile.close();
    uint64_t totalBytes = (uint64_t)NUM_INTS * sizeof(int32_t);

    std::cout << "Successfully created " << FILE_NAME << " ("
        << (totalBytes / (1024.0 * 1024.0))
        << " MB)" << std::endl;
    std::cout << "Expected Sum: " << (long long)NUM_INTS * VALUE << std::endl;

    return 0;
}