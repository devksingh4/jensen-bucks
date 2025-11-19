# Create the 1 billion row file as measurements.txt (~14GB)
INPUT="../data/measurements_full.txt"
# Compile and run the c++ baseline.
g++ -o base -O2 base.cpp
time ./base $INPUT # ~17 mins

# Compile and run my cuda solution.
nvcc -o fast -O2 fast.cu
time ./fast $INPUT 3584 6000000  # ~17s on V100

diff cuda_measurements.out measurements.out
