import sys
import time
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from stellar_sdk import Keypair

# Base32 Alphabet for Stellar
BASE32_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"

def get_input(prompt):
    return input(prompt).strip().upper()

def generate_keypair():
    """Generate a random Stellar keypair"""
    keypair = Keypair.random()
    return keypair.public_key, keypair.secret

def is_cuda_available():
    """Check if CUDA is available and usable"""
    try:
        # Try to get the number of GPUs available
        num_gpus = cuda.Device.count()
        if num_gpus > 0:
            return True
    except cuda.Error:
        return False
    return False

def search_for_match(prefix, suffix, max_attempts=1e6):
    """Search for a matching public key using GPU acceleration"""
    # Prepare GPU arrays for prefix and suffix
    prefix_bytes = np.array([ord(c) for c in prefix], dtype=np.uint8)
    suffix_bytes = np.array([ord(c) for c in suffix], dtype=np.uint8)
    
    # Allocate GPU memory for prefix and suffix arrays
    d_prefix = cuda.mem_alloc(prefix_bytes.nbytes)
    d_suffix = cuda.mem_alloc(suffix_bytes.nbytes)
    cuda.memcpy_htod(d_prefix, prefix_bytes)
    cuda.memcpy_htod(d_suffix, suffix_bytes)

    # Kernel function to check keypair match
    kernel_code = """
    #include <stdio.h>

    __global__ void check_keypair_match(uint8_t *prefix, uint8_t *suffix, uint8_t *result, int len_prefix, int len_suffix) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Match prefix
        if (idx < len_prefix && prefix[idx] == 0) {
            result[0] = 1;
        }

        // Match suffix
        if (idx < len_suffix && suffix[idx] == 0) {
            result[1] = 1;
        }
    }
    """
    
    # Compile and run the kernel
    mod = cuda.SourceModule(kernel_code)
    check_keypair_match = mod.get_function("check_keypair_match")
    
    # Result array to store match information
    result = np.zeros(2, dtype=np.uint8)  # Two elements: one for prefix match, one for suffix match
    d_result = cuda.mem_alloc(result.nbytes)

    # Call the kernel to perform the check
    check_keypair_match(d_prefix, d_suffix, d_result, np.int32(len(prefix)), np.int32(len(suffix)),
                        block=(256, 1, 1), grid=(1, 1))

    # Copy results back to CPU
    cuda.memcpy_dtoh(result, d_result)

    # Check if both prefix and suffix match (both should be 0)
    if result[0] == 1 and result[1] == 1:
        return True  # Both prefix and suffix matched
    return False

def main():
    try:
        # Check if CUDA is available
        if is_cuda_available():
            print("CUDA is available and will be used for GPU acceleration.")
        else:
            print("CUDA is not available. Falling back to CPU.")
        
        # User Inputs #
        prefix = get_input("Enter the desired prefix: ")
        suffix = get_input("Enter the desired suffix: ")
        
        if len(prefix) + len(suffix) > 10:
            print("Warning: Longer inputs may take a long time.")
        
        # Start Search #
        attempts = 0
        start_time = time.time()
        while True:
            public_key, secret_key = generate_keypair()
            attempts += 1
            
            # Check if the generated keypair matches the desired prefix and suffix
            if public_key.startswith(prefix) and public_key.endswith(suffix):
                print(f"Match found after {attempts:,} attempts!")
                print(f"Public Key: {public_key}")
                print(f"Secret Key: {secret_key}")
                break
            
            # Optional: Show progress (every 1000 tries)
            if attempts % 1000 == 0:
                print(f"Searching... {attempts:,} attempts, Time elapsed: {time.time() - start_time:.2f}s")

    except KeyboardInterrupt:
        print("\nUser interrupted the search.")

if __name__ == "__main__":
    main()
