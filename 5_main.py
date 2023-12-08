import os
import numpy as np
import pylab as plt
import polars as pl
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
truncate_data_to_time_s = 1e9 # for faster triggering use a smaller value
# for full analysis use a ridiculously large value like 1e9
# script will auto re-use triggering if all other parameters are identical, so you only need
# to wait once

# inputs
# pulse selection quantities
min_time_since_last_s = 0.5
min_time_to_next_s = 0.010
min_time_since_last_s_for_average_pulse = 0.6

# filter inputs
nsamples = 1000
npre = 500
filter_orthogonal_to_exponential_time_constant_ms = 25
filter_pretrigger_ignore_samples = 10 # this value to exclude rising samples from pretrigger period

# pulse classification inputs
max_residual_rms = 60
# spikeyness_threshold_foil_plus_non_foil = 7.7e-4
# spikeyness_threshold_non_foil_only = 17e-3
spikeyness_threshold_foil_plus_non_foil = 0.0006
spikeyness_threshold_non_foil_only = 0.006
frontload_n_samples = 100

# energy calibration inputs
am241_Q_eV = 5637.82e3

# hist inputs
roi_lo = 5.4e6
roi_hi = 6.0e6


analyzer = truebq_analysis.NpyAnalyzer(fname_npy, polarity, npre, nsamples)
analyzer.trigger_pulses_or_load_from_cache(trigger_threshold=trigger_threshold, 
                                           trigger_filter=trigger_filter,
                                           truncate_data_to_time_s=truncate_data_to_time_s)
noise = analyzer.calculate_noise_triggers(nsamples, noise_n_dead_samples_after_previous_pulse)
noise.plot(f"noise record length {nsamples=}")
noise_long = analyzer.calculate_noise_triggers(noise_long_n_samples, noise_n_dead_samples_after_previous_pulse, max_triggers=200)
noise_long.plot(f"noise long {noise_long_n_samples=}")
analyzer.calculate_spikeyness_pretrig_mean_pulse_rms(frontload_n_samples)
analyzer.spikeyness_debug_plot(min_time_since_last_s=min_time_since_last_s,
                               min_time_to_next_s=min_time_to_next_s,
                               spikeyness_threshold_foil_plus_non_foil=spikeyness_threshold_foil_plus_non_foil,
                               spikeyness_threshold_non_foil_only=spikeyness_threshold_non_foil_only)
avg_pulse_obj = analyzer.calculate_average_pulse(min_time_since_last_s_for_average_pulse, min_time_to_next_s, 
                                                 spikeyness_threshold_foil_plus_non_foil)
avg_pulse_obj.plot()
analyzer.calculate_filter(avg_pulse_obj.values(), noise.autocorr(), noise.spectrum.spectrum(), 
                          filter_pretrigger_ignore_samples,
                          filter_orthogonal_to_exponential_time_constant_ms)
analyzer.filter()
# max_residual_rms = npyfilter.mad_threshold(analyzer.df["residual_rms"].to_numpy(), n_sigma=residual_rms_n_sigma)
class_meaning = analyzer.classify(min_time_since_last_s=min_time_since_last_s, 
                                  min_time_to_next_s=min_time_to_next_s,
                                  spikeyness_threshold_foil_plus_non_foil=spikeyness_threshold_foil_plus_non_foil,
                                  spikeyness_threshold_non_foil_only=spikeyness_threshold_non_foil_only,
                                  max_residual_rms=max_residual_rms)
analyzer.classification_debug_plots(class_meaning)
analyzer.filt_value_to_energy_with_ptmean_correlation_removal(median_energy=am241_Q_eV)
analyzer.df.write_parquet("out.parquet")


print(class_meaning)
df=analyzer.df[["trig_ind","classification","energy","time_since_last_s"]]
energy = df["energy"].to_numpy(writable=True)
classification=df["classification"].to_numpy()
for c, meaning in class_meaning.items():
    print(f"{c=} {meaning} {np.sum(classification==c)}")
    if c>0:
        energy[classification==c]=-c
energy[classification==1]=6e6
df=df.with_columns(energy_classified=energy)
bin_edges = np.arange(0,6.1e6,1000)
counts, _ = np.histogram(df["energy_classified"].to_numpy(), bin_edges)
live_time_s = (df.filter(pl.col("energy_classified")>0)["time_since_last_s"].to_numpy()-min_time_since_last_s).sum()
total_counts = counts.sum()
total_activity = total_counts/live_time_s
total_activity_uncertainty = np.sqrt(total_counts)/live_time_s
inds_roi = np.nonzero((energy>roi_lo) & (energy<roi_hi))[0]
total_counts_roi = len(inds_roi)
am241_bq_roi_bq = total_counts_roi/live_time_s
am241_bq_roi_bq_sigma = np.sqrt(total_counts_roi)/live_time_s
def binsize(x):
    return x[1]-x[0]
def midpoints(x):
    return (x[1:]+x[:-1])/2
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
plt.yscale("log")
plt.tight_layout()






