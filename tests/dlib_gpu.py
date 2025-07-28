import dlib

# Check if dlib was compiled with CUDA support
print(f"dlib was compiled with CUDA support: {dlib.DLIB_USE_CUDA}")

# Check the number of available CUDA devices
if dlib.DLIB_USE_CUDA:
    print(f"Number of available GPUs: {dlib.cuda.get_num_devices()}")