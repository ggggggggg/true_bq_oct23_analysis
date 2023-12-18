import numpy as np
import ljhfiles
import os

fname_ljh = os.path.join("/home/pcuser/data","20231003","0002","20231003_run0002_chan3.ljh")
ljh = ljhfiles.LJHFile(fname_ljh)
ljh.write_as_npy(f"{fname_ljh}.npy")


