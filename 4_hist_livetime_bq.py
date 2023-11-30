import os
import numpy as np
import pylab as plt
import pickle
import npyfilter
import mass
import scipy
plt.ion()
plt.close("all")

# inputs
min_time_since_last_s = 0.2

fname_ljh = os.path.join(".","20231003","0002","20231003_run0002_chan3.ljh")
fname_npy = f"{fname_ljh}.npy"
fname_header = f"{fname_npy}.header"
fname_trig_inds = f"{fname_npy}.trig_inds.npz"
fname_energy = f"{fname_npy}.energy.npy"
with open(fname_header,"rb") as f:
    header = pickle.load(f)
    frametime_s = header["timebase_s"]
data = np.load(fname_npy, mmap_mode="r").reshape(-1)
energy = np.load(fname_energy, mmap_mode="r")
trig_data = np.load(fname_trig_inds)
trig_inds = trig_data["trig_inds"]
noise_inds = trig_data["noise_inds"]
noise_long_inds = trig_data["noise_long_inds"]



# select well isolated pulses to create median pulse
time_since_last_s = np.diff(trig_inds*frametime_s, prepend=trig_inds[0])
time_to_next_s = np.diff(trig_inds*frametime_s, append=trig_inds[-1])

bin_edges = np.arange(0,6e6,2000)
roi_lo = 5537000
roi_hi = 5665000
inds_roi = np.nonzero((energy>roi_lo) & (energy<roi_hi))[0]
total_counts_roi = len(inds_roi)
live_time_s = np.sum(time_since_last_s[energy>0]-min_time_since_last_s)
#   5. Calc Bq
am241_bq_roi_bq = total_counts_roi/live_time_s
am241_bq_roi_bq_sigma = np.sqrt(total_counts_roi)/live_time_s
# 	4. Plot histogram
# live_time_s = live_time_algo_and_sim.live_time_from_live_ranges(live_ranges_s)
def binsize(x):
    return x[1]-x[0]
def midpoints(x):
    return (x[1:]+x[:-1])/2
counts, _ = np.histogram(energy, bin_edges)
total_counts = counts.sum()
total_activity = total_counts/live_time_s
total_activity_uncertainty = np.sqrt(total_counts)/live_time_s
plt.figure()
plt.plot(midpoints(bin_edges), counts, drawstyle="steps-mid", label="live spectrum")
roi_plot_inds = (midpoints(bin_edges) > roi_lo) & (midpoints(bin_edges)<roi_hi)
plt.plot(midpoints(bin_edges)[roi_plot_inds], counts[roi_plot_inds],"r", drawstyle="steps-mid", label="Am241 ROI")
plt.fill_between(midpoints(bin_edges)[roi_plot_inds], counts[roi_plot_inds], step="mid", color="r", alpha=0.5)
plt.xlabel("energy / eV (with category having specific values)")
plt.ylabel(f"counts per {binsize(bin_edges):0.1f} eV")
plt.title(f"""Histogram Live Time = {live_time_s:0.2f} s. Total Counts = {total_counts}
          Total activity = {total_activity:.3f}+/-{total_activity_uncertainty:.3f} events/s, non ROI counts = {total_counts-total_counts_roi}
          ROI counts = {total_counts_roi=}
          ROI activity = {am241_bq_roi_bq:0.3f}+/-{am241_bq_roi_bq_sigma:0.3f}""")
plt.legend()
plt.tight_layout()