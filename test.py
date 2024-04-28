import torch

# Check if GPU is available
if torch.cuda.is_available():
    # Get the CUDA device count
    device_count = torch.cuda.device_count()
    print("Number of available GPUs:", device_count)
    
    # Iterate through each GPU and print memory information
    for i in range(device_count):
        gpu_properties = torch.cuda.get_device_properties(i)
        print(f"GPU {i}:")
        print(f"  Name: {gpu_properties.name}")
        print(f"  Total Memory: {gpu_properties.total_memory / (1024**2)} MB")
else:
    print("CUDA is not available. Check if you have selected a GPU accelerator in the Kaggle notebook settings.")

