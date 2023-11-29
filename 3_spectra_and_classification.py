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
min_time_since_last_s = 0.3
min_time_to_next_s = 0.005
# pulse lengths
nsamples = 1000
npre = 500
pretrigger_ignore_samples = 10 # this value to exclude rising samples from pretrigger period
polarity = -1
spikeyness_threshold_foil_plus_non_foil = 45
spikeyness_threshold_non_foil_only = 100
spikeness_last_val_offset = 100
filter_orthogonal_to_exponential_time_constant_ms = 2.0

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
    spikeyness[i] = (peakval-lastval)
    pulse_rms[i] = np.sqrt(np.sum((pulse[npre:]-pretrig_mean[i])**2))

plt.hist(spikeyness, np.linspace(0,200,100))
plt.xlabel("spikeyness")
plt.ylabel("number of occurences")
plt.axvline(spikeyness_threshold_non_foil_only, label="spikeyness_threshold_non_foil_only",color="r")
plt.axvline(spikeyness_threshold_foil_plus_non_foil, label="spikeness_threshold_foil_plus_non_foil",color="r")
plt.legend()

npyfilter.plot_inds(data, npre, nsamples, trig_inds[(spikeyness>spikeyness_threshold_foil_plus_non_foil)&(isolated_bool)],f"spikeyness>{spikeyness_threshold_foil_plus_non_foil:.2f}")
npyfilter.plot_inds(data, npre, nsamples, trig_inds[(spikeyness<0.03)],f"spikeyness<0.03")

def gather_pulse_from_inds(data, npre, nsamples, inds):
    pulses = np.zeros((nsamples, len(inds)))
    for i, ind in enumerate(inds):
        pulses[:,i] = data[ind-npre:ind+nsamples-npre]
    return pulses

clean_inds = trig_inds[(spikeyness<spikeyness_threshold_foil_plus_non_foil)&(isolated_bool)]

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


# def autocorrelation_broken_from_pulses(noise_pulses):
#     """noise_pulses[:,0] is the first pulse"""
#     nsamples, npulses = noise_pulses.shape
#     records_used = samples_used = 0
#     ac = np.zeros(nsamples, dtype=float)

#     for i in range(npulses):
#         pulse = noise_pulses[:,i]
#         pulse -= pulse.mean()
#         ac += np.correlate(pulse, pulse, 'full')[nsamples-1:]

#     ac /= npulses
#     ac /= nsamples - np.arange(nsamples, dtype=float)
#     return ac


spectrum=npyfilter.spectrum_from_pulse(noise_pulses, frametime_s)
psd = spectrum.spectrum()
spectrum.plot()
noise_autocorr = mass.mathstat.power_spectrum.autocorrelation_broken_from_pulses(noise_pulses)
filter_obj = mass.ExperimentalFilter(avg_clean_pulse, npre-pretrigger_ignore_samples,
                                 psd, sample_time=frametime_s, 
                                 noise_autocorr=noise_autocorr,
                                 tau=filter_orthogonal_to_exponential_time_constant_ms)
filter_obj.compute()
filter_obj.report()
chosen_filter = filter_obj.filt_noexpcon

filt_value = np.zeros(len(trig_inds))
residual_rms = np.zeros(len(trig_inds))
filt_value_template = np.zeros(len(trig_inds))
template = avg_clean_pulse-np.mean(avg_clean_pulse)
template = template/np.sqrt(np.dot(template, template))
for i in range(len(trig_inds)):
    j = trig_inds[i]
    pulse = data[j-npre:j+nsamples-npre]*polarity
    filt_value[i] = np.dot(chosen_filter, pulse)
    filt_value_template[i] = np.dot(template, pulse)
    residual = pulse-template*filt_value_template[i]
    residual_std_dev = np.std(residual)
    residual_rms[i] = residual_std_dev  

def median_absolute_deviation(x):
    med = np.median(x)
    return np.median(np.abs(x-med))
def mad_threshold(x, n_sigma=5):
    med = np.median(x)
    mad = median_absolute_deviation(x)
    sigma = mad*1.4826
    return med+5*sigma

max_residual_rms = mad_threshold(residual_rms)

classification_meaning = {0: "foil clean",
                          1: "foil + non_foil",
                          2: "non_foil_only",
                          3: "last too close",
                          4: "next too close",
                          5: "last and next too close",
                          6: "high residual_rms"}
classification = np.zeros(len(trig_inds), dtype=int)
classification[(classification==0)&(spikeyness>spikeyness_threshold_non_foil_only)]=2
classification[(classification==0)&(time_since_last_s<min_time_since_last_s)&(time_to_next_s<min_time_to_next_s)]=5
classification[(classification==0)&(time_since_last_s<min_time_since_last_s)]=3
classification[(classification==0)&(time_to_next_s<min_time_to_next_s)]=4
classification[(classification==0)&(spikeyness>spikeyness_threshold_foil_plus_non_foil)]=1
classification[(classification==0)&(residual_rms>max_residual_rms)]=1

# plot representatives of pulse classifications
for c in range(len(classification_meaning)):
    inds = np.nonzero(classification==c)[0]
    c_meaning = classification_meaning[c]
    print(f"{c=} {c_meaning} {len(inds)=}")
    npyfilter.plot_inds(data, npre, nsamples, trig_inds[inds], label=f"{c=} {c_meaning}", max_pulses_to_plot=50)
    
