import os
import sys
from my_ingestion import run_model
import time

SRC_DIR = os.path.dirname(os.path.abspath(__file__))        
BENCHMARK_CACHE_DIR = os.path.join(SRC_DIR, "..", "Benchmarks")             # Path to the benchmark folder. If the file does not exist, it will be created there.

def run(model_name):
    model_path = os.path.join(SRC_DIR, "models", model_name)

    print("\n=====================================================")
    print("Starting the model training and evaluation process...")
    print("=====================================================")
    print("start time ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "\n")
    start = time.time()

    # training & evaluating the model
    run_model(SRC_DIR, model_path, regime="scarce", benchmark_dir=BENCHMARK_CACHE_DIR)

    end = time.time()
    print("\n===============================================")
    print("Finished! Execution time: ", (end - start)//60, " minutes\n")
    print("===============================================")


if __name__ == "__main__":
    # read input in command line
    model_name = sys.argv[1]
    
    print(f"Using model: {model_name}")
    if not os.path.exists(os.path.join(SRC_DIR, "models", model_name)):
        raise ValueError(f"Model name {model_name} does not exist, provide a valid model name e.g. <bi_transformer>")

    run(model_name)
