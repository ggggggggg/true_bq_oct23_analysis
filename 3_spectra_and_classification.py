import os
import numpy as np
import pylab as plt
import pickle
import npyfilter
import mass
plt.ion()
plt.close("all")

# inputs
# pulse selection quantities for average pulse
min_time_since_last_s = 0.4
min_time_to_next_s = 0.1
# pulse lengths
nsamples = 1000
npre = 500
pretrigger_ignore_samples = 10 # this value to exclude rising samples from pretrigger period
polarity = -1
spikeyness_threshold = 0.06
spikeness_last_val_offset = 100

fname_ljh = os.path.join(".","20231003","0002","20231003_run0002_chan3.ljh")
fname_npy = f"{fname_ljh}.npy"
fname_header = f"{fname_npy}.header"
fname_trig_inds = f"{fname_npy}.trig_inds.npz"
with open(fname_header,"rb") as f:
    header = pickle.load(f)
    frametime_s = header["timebase_s"]
data = np.load(fname_npy, mmap_mode="r").reshape(-1)
trig_data = np.load(fname_trig_inds)
trig_inds = trig_data["trig_inds"]
noise_inds = trig_data["noise_inds"]
noise_long_inds = trig_data["noise_long_inds"]



# select well isolated pulses to create median pulse
time_since_last_s = np.diff(trig_inds*frametime_s, prepend=trig_inds[0])
time_to_next_s = np.diff(trig_inds*frametime_s, append=trig_inds[-1])
isolated_bool = (time_since_last_s>min_time_since_last_s)&(time_to_next_s>min_time_to_next_s)

# calculate some per pulse quantities
spikeyness = np.zeros(len(trig_inds))
pulse_rms = np.zeros(len(trig_inds))
pretrig_mean = np.zeros(len(trig_inds))
for i in range(len(trig_inds)):
    j = trig_inds[i]
    pulse = data[j-npre:j+nsamples-npre]*polarity
    pretrig_mean[i] = np.mean(pulse[npre//2])
    lastval = np.abs(pulse[npre+spikeness_last_val_offset]-pretrig_mean[i])
    peakval = np.amax(pulse)-pretrig_mean[i]
    spikeyness[i] = (peakval-lastval)/lastval
    pulse_rms[i] = np.sqrt(np.sum((pulse[npre:]-pretrig_mean[i])**2))

plt.hist(spikeyness, np.arange(0,1,0.001))
plt.xlabel("spikeyness")
plt.ylabel("number of occurences")
plt.axvline(spikeyness_threshold, label="spikeness_threshold",color="r")
plt.legend()

npyfilter.plot_inds(data, npre, nsamples, trig_inds[(spikeyness>spikeyness_threshold)&(isolated_bool)],f"spikeyness>{spikeyness_threshold:.2f}")
npyfilter.plot_inds(data, npre, nsamples, trig_inds[(spikeyness<0.03)],f"spikeyness<0.03")

def gather_pulse_from_inds(data, npre, nsamples, inds):
    pulses = np.zeros((nsamples, len(inds)))
    for i, ind in enumerate(inds):
        pulses[:,i] = data[ind-npre:ind+nsamples-npre]
    return pulses

clean_inds = trig_inds[(spikeyness<spikeyness_threshold)&(isolated_bool)]

npyfilter.plot_inds(data, npre, nsamples, clean_inds,f"clean_inds")
clean_pulses = gather_pulse_from_inds(data, npre, nsamples, clean_inds)
noise_pulses = gather_pulse_from_inds(data, npre, nsamples, noise_inds)
avg_clean_pulse = np.mean(clean_pulses, axis=1)
plt.figure()
plt.plot(clean_pulses[:,:200])
plt.plot(avg_clean_pulse,"k", label="avg clean pulse")
plt.legend()
plt.title("200 clean pulses")
plt.figure()
plt.plot(noise_pulses[:,:200])
plt.title("200 noise pulses")


spectrum=npyfilter.spectrum_from_pulse(noise_pulses, frametime_s)
f = spectrum.frequencies()
psd = spectrum.spectrum()
filter = mass.Filter(avg_clean_pulse, npre,
                                 self.noise_psd[...],
                                 self.noise_autocorr, sample_time=self.timebase,
                                 shorten=shorten)