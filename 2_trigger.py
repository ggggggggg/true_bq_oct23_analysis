import os
import numpy as np
import pylab as plt
import pickle
import npyfilter
plt.ion()
plt.close("all")

fname_ljh = os.path.join(".","20231003","0002","20231003_run0002_chan3.ljh")
fname_npy = f"{fname_ljh}.npy"
fname_header = f"{fname_npy}.header"
polarity = -1 # postive for positive oging pulses, negative for negative going
trigger_filter = np.array([-1]*10+[+1]*10,dtype=float) * polarity # negate for negative pulses
trigger_threshold = 50
noise_n_dead_samples_after_previous_pulse = 70000
noise_n_samples = 1000
noise_long_n_samples = 100000
truncate_data_to_time_s = 1e3

fname_trig_inds = f"{fname_npy}.trig_inds.npz"

with open(fname_header,"rb") as f:
    header = pickle.load(f)
    frametime_s = header["timebase_s"]
trunacate_to_frame = int(truncate_data_to_time_s/frametime_s)
data_raw = np.load(fname_npy, mmap_mode="r").reshape(-1)
#truncate for faster analysis
data = data_raw[:min(len(data_raw), trunacate_to_frame)]



trig_inds = npyfilter.fasttrig_filter_trigger(data, trigger_filter[:], trigger_threshold)
noise_inds = npyfilter.get_noise_trigger_inds(trig_inds, n_dead_samples_after_previous_pulse=noise_n_dead_samples_after_previous_pulse, 
                           n_record_samples=noise_n_samples, max_noise_triggers=10000)
noise_long_inds = npyfilter.get_noise_trigger_inds(trig_inds, n_dead_samples_after_previous_pulse=noise_n_dead_samples_after_previous_pulse, 
                           n_record_samples=noise_long_n_samples, max_noise_triggers=len(trig_inds))

# save triggers to file
np.savez(fname_trig_inds, trig_inds=trig_inds, noise_inds=noise_inds, noise_long_inds=noise_long_inds)


# debug plots
plt.figure()
plt.plot(noise_long_inds*frametime_s,data[noise_long_inds],".")
plt.xlabel("time (s)")
plt.ylabel("value at start of noise record (like pt mean)")

skip = 10
N = 100000
lo = N*10
hi = lo+N*skip
trig_inds_plot = trig_inds[(trig_inds>lo) & (trig_inds < hi)]
noise_inds_plot = noise_inds[(noise_inds>lo) & (noise_inds < hi)]
noise_long_inds_plot = noise_long_inds[(noise_long_inds>lo) & (noise_long_inds < hi)]
plt.figure()
plt.plot(np.arange(lo, hi, skip)*frametime_s/3600, data[lo:hi:skip])
plt.plot(trig_inds_plot*frametime_s/3600, data[trig_inds_plot],"ro",label="trig_inds")
plt.plot(noise_inds_plot*frametime_s/3600, data[noise_inds_plot],"bx",label="noise_inds")
plt.plot(noise_long_inds_plot*frametime_s/3600, data[noise_long_inds_plot],"cv",label="noise_long_inds")
plt.xlabel("time (hour)")
plt.legend()
plt.ylabel("signal (arbs)")
plt.tight_layout()
plt.grid(True)
