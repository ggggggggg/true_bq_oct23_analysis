import os
import numpy as np
import pathlib

dir_path = os.path.dirname(os.path.realpath(__file__))
cache_dir=pathlib.Path(os.path.join(dir_path,"_cache"))
cache_dir.mkdir(exist_ok=True) 
verbose = True

def get_fname(key: str):
    return os.path.join(cache_dir, key)
