import time
import cudf as pd

run_times = []

for i in range(3):
    start = time.time()
    
    df = pd.read_csv(
        "/home/dsingh/source/devksingh4/jensen-bucks/data/measurements_full.txt",
        sep=';',
        header=None,
        names=["station", "measure"]
    )
    df = df.groupby("station").agg(["min", "max", "mean"])
    df = df.sort_values("station")
    
    end = time.time()
    elapsed = end - start
    run_times.append(elapsed)
    print(f"Run {i+1}: {elapsed:.4f} seconds")

print(f"\nMean: {sum(run_times)/len(run_times):.4f} seconds")
print(f"Min: {min(run_times):.4f} seconds")
print(f"Max: {max(run_times):.4f} seconds")