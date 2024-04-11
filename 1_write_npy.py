import numpy as np
import ljhfiles
import os

y_dir = "/home/pcuser/data"
my_folder = "20240314"
my_runnum = "0000"
my_chan = "3" # to do : implement make this "*" for processing all channels


fname_ljh = os.path.join("/home/pcuser/data","20240314","0000","20240314_run0000_chan3.ljh")
ljh = ljhfiles.LJHFile(fname_ljh)
ljh.write_as_npy(f"{fname_ljh}.npy")


