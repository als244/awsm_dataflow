import ctypes
_cudart = ctypes.CDLL('libcudart.so')

def pin_tensor(tensor):
    size_bytes = int(tensor.numel() * tensor.element_size())
    if tensor.device.type == "cpu":
        ret = _cudart.cudaHostRegister(ctypes.c_void_p(tensor.data_ptr()), ctypes.c_size_t(size_bytes), ctypes.c_uint(0))
        if ret != 0:
            print(f"cudaHostRegister failed: {ret}; for tensor: {tensor}, with data_ptr of: {tensor.data_ptr()} and size of: {size_bytes}")
    return tensor

def destory_tensor(tensor, is_pinned=True):

    size_bytes = int(tensor.numel() * tensor.element_size())

    if tensor.device.type == "cpu" and is_pinned:
        ret = _cudart.cudaHostUnregister(ctypes.c_void_p(tensor.data_ptr()))
        if ret != 0:
            print(f"Failed to unregsiter cpu tensor at addr: {tensor.data_ptr()} of size {size_bytes}")
    
    del tensor