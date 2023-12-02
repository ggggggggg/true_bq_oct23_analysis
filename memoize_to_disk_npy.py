import os
import numpy as np
import pathlib

dir_path = os.path.dirname(os.path.realpath(__file__))
cache_dir=pathlib.Path(os.path.join(dir_path,"_cache"))
cache_dir.mkdir(exist_ok=True) 
verbose = True

def get_fname(key: str):
    return os.path.join(cache_dir, key)


# def persist_to_file_npy(original_func):
#     def decorator(original_func):
#         def new_func(*args, **kwargs):
#             key = (args,kwargs)
#             fname = os.path.join(cache_dir, str(hash(key)))
#             try:
#                 array = np.load(fname)
#                 print(f"cache hit for {fname}")
#             except FileNotFoundError:
#                 print(f"cache miss for {fname}")
#                 array = original_func(*args, **kwargs)
#                 np.save(fname, array)
#         return new_func

#     return decorator(original_func)