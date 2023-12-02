import os
import numpy as np
import pylab as plt
import pickle
import npyfilter
import scipy
import truebq_analysis
plt.ion()
plt.close("all")


fname_npy = os.path.join(".","20231003","0002","20231003_run0002_chan3.ljh.npy")
polarity = -1 # postive for positive oging pulses, negative for negative going
trigger_filter = np.array([-1]*10+[+1]*10,dtype=float) * polarity # negate for negative pulses
trigger_threshold = 200

noise_n_dead_samples_after_previous_pulse = 70000
noise_long_n_samples = 100000
truncate_data_to_time_s = 1e3 # for faster triggering use a smaller value
# for full analysis use a ridiculously large value like 1e9
# script will auto re-use triggering if all other parameters are identical, so you only need
# to wait once

# inputs
# pulse selection quantities
min_time_since_last_s = 0.15
min_time_to_next_s = 0.005
min_time_since_last_s_for_average_pulse = 0.6

# pulse length inputs
nsamples = 1000
npre = 500
pretrigger_ignore_samples = 10 # this value to exclude rising samples from pretrigger period
# determine from average pulse plot

# pulse classification inputs
residual_rms_n_sigma=15
spikeyness_threshold_foil_plus_non_foil = 7.7e-4
spikeyness_threshold_non_foil_only = 17e-3
frontload_n_samples = 150
filter_orthogonal_to_exponential_time_constant_ms = 25

# energy calibration inputs
am241_Q_eV = 5637.82e3


analyzer = truebq_analysis.NpyAnalyzer(fname_npy, polarity, npre, nsamples)
analyzer.trigger_pulses_or_load_from_cache(trigger_threshold=trigger_threshold, 
                                           trigger_filter=trigger_filter,
                                           truncate_data_to_time_s=truncate_data_to_time_s)
noise = analyzer.calculate_noise_triggers(nsamples, noise_n_dead_samples_after_previous_pulse)
noise.plot()
noise_long = analyzer.calculate_noise_triggers(noise_long_n_samples, noise_n_dead_samples_after_previous_pulse)
noise_long.plot()
analyzer.calculate_spikeyness_pretrig_mean_pulse_rms(frontload_n_samples)
analyzer.spikeyness_debug_plot(min_time_since_last_s=min_time_since_last_s,
                               min_time_to_next_s=min_time_to_next_s,
                               spikeyness_threshold_foil_plus_non_foil=spikeyness_threshold_foil_plus_non_foil,
                               spikeyness_threshold_non_foil_only=spikeyness_threshold_non_foil_only)
avg_pulse_obj = analyzer.calculate_average_pulse(min_time_since_last_s_for_average_pulse, min_time_to_next_s, 
                                                 spikeyness_threshold_foil_plus_non_foil)
avg_pulse_obj.plot()
analyzer.calculate_filter(avg_pulse_obj.values(), noise.autocorr(), noise.spectrum.psd(), 
                          filter_orthogonal_to_exponential_time_constant_ms)
analyzer.filter()
max_residual_rms = npyfilter.mad_threshold(analyzer.df["residual_rms"].to_numpy(), n_sigma=residual_rms_n_sigma)
class_meaning = analyzer.classify(min_time_since_last_s=min_time_to_next_s, 
                                  min_time_to_next_s=min_time_to_next_s,
                                  spikeyness_threshold_foil_plus_non_foil=spikeyness_threshold_foil_plus_non_foil,
                                  spikeyness_threshold_non_foil_only=spikeyness_threshold_non_foil_only,
                                  max_residual_rms=max_residual_rms)
analyzer.filt_value_to_energy_with_ptmean_correlation_removal(median_energy=am241_Q_eV)
analyzer.df.write_parquet("out.parquet")



