import time
import torch  # PyTorch library
import os
import sys
print(f"\nPython Version :: \n{sys.version}\n{'-' * 120}")
print(f"Nvidia CUDA compiler :: \n{os.system('nvcc --version')}\n{'-' * 120}")

# Running a simple operation on Nvidia Devices
def validate_gpu(device_type: int = -1) -> None:
    """
    Function to perform a tensor operation on either a GPU or CPU based on the specified device_type.

    :param device_type: Integer representing the GPU device index to use. If negative, the computation will run on the CPU.
    :return: None
    """
    device = None

    start_time = time.time() # Timing the tensor operations
    if device_type >= 0:
        # By default, tensors run on CPU, and have to be manually switched to GPU to utilize GPU computation power
        # This can be controlled via device = ? parameter
        device = torch.device(f'cuda:{device_type}')
        print(f"Using GPU device {device_type}: {torch.cuda.get_device_name(device_type)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Create large random tensors and move them to the specified device
    vector_a = torch.rand((50000, 50000), dtype=torch.float32).to(device) # 2.5 billion elements in matrix
    vector_b = torch.rand((50000, 50000), dtype=torch.float32).to(device) # 2.5 billion elements in matrix
    result = torch.matmul(vector_a, vector_b)

    end_time = time.time() # End time for tensor operations

    # Determine the device on which the result tensor is computed
    tensor_result_device = 'CPU' if device_type < 0 else torch.cuda.get_device_name(device_type)

    print(f"Tensor a is on device: {vector_a.device}")
    print(f"Tensor b is on device: {vector_b.device}")
    print(f"\n{result}\n")
    print(f"Result tensor is computed on :: {tensor_result_device}\n\n{'*' * 140}")
    print(f"Tensor operations took {end_time - start_time:.4f} seconds.\n{'*' * 140}")
    # Sample output:
    # CPU: Tensor operations took 187.4379 seconds.
    # GPU: Tensor operations took 16.4074 seconds

# Check if CUDA (GPU support) is available
_is_gpu_available = torch.cuda.is_available() # Toggle GPU vs CPU here

if _is_gpu_available:
    # If GPU(s) are available, print the number of GPUs and perform operations on each
    gpu_count = torch.cuda.device_count()
    print(f"GPU found for this system\nNumber of GPU's found :: {gpu_count}\n{'-' * 120}\n")
    for gpu_device_index in range(gpu_count):
        print(f"Device {gpu_device_index} :: {str(torch.cuda.get_device_name(gpu_device_index))}\n")
        validate_gpu(gpu_device_index)
else:
    # If no GPU is found, perform the operation on the CPU
    print("GPU not found on this system")
    validate_gpu()