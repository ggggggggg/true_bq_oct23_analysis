import os
import numpy as np
import pylab as plt
import pickle
import npyfilter
import scipy
import memoize_to_disk_npy
import polars as pl
import hashlib
import mass
am241_Q_eV = 5637.82e3
class NpyNoise():
    def __init__(self, title, data, noise_trig_inds, frametime_s, nsamples):
        self.title = title
        self.data = data
        self.noise_trig_inds = noise_trig_inds
        self.frametime_s = frametime_s
        self.nsamples = nsamples
        self.spectrum = self._calc()

    def plot(self, title):
        ret = self.spectrum.plot()
        plt.title(title)
        return ret
    
    def _calc(self):
        pulses = npyfilter.gather_pulse_from_inds(self.data, 0, self.nsamples, self.noise_trig_inds)
        spectrum=npyfilter.spectrum_from_pulse(pulses, self.frametime_s)
        return spectrum
    
    def autocorr(self):
        pulses = npyfilter.gather_pulse_from_inds(self.data, 0, self.nsamples, self.noise_trig_inds)
        noise_autocorr = mass.mathstat.power_spectrum.autocorrelation_broken_from_pulses(pulses)
        return noise_autocorr


class NpyAveragePulse():
    def __init__(self, title, inds_used, data, npre, nsamples):
        self.title=title
        self.inds_used = inds_used
        self.data = data
        self.npre = npre
        self.nsamples = nsamples

    def plot(self):
        pulses = npyfilter.gather_pulse_from_inds(self.data, self.npre, self.nsamples, self.inds_used)
        avg_pulse = np.mean(pulses, axis=1)
        plt.figure()
        plt.plot(pulses[:,:200])
        plt.plot(avg_pulse,"k", label="avg pulse")
        plt.legend()

    def values(self):
        pulses = npyfilter.gather_pulse_from_inds(self.data, self.npre, self.nsamples, self.inds_used)
        return np.mean(pulses, axis=1)




class NpyAnalyzer():
    def __init__(self, fname_npy, pulse_polarity, npre, nsamples):
        self.fname_npy = fname_npy
        self.data = data_raw = np.load(fname_npy, mmap_mode="r").reshape(-1)
        fname_header = f"{fname_npy}.header"
        with open(fname_header,"rb") as f:
            header = pickle.load(f)
            self.frametime_s = header["timebase_s"]
        self.pulse_polarity=pulse_polarity
        self.df = pl.DataFrame()
        self.npre = npre
        self.nsamples = nsamples

    def _trigger_pulses_or_load_from_cache(self, trigger_filter, trigger_threshold, truncate_data_to_time_s):
        key = f"{hashlib.sha256((str((str(trigger_filter), trigger_threshold, truncate_data_to_time_s, self.fname_npy))).encode()).hexdigest()}.npy"
        # key = f"{hashlib.sha256((str((str(trigger_filter), trigger_threshold, truncate_data_to_time_s))).encode()).hexdigest()}.npy"
        fname = memoize_to_disk_npy.get_fname(key)
        try:
            trig_inds = np.load(fname)
            print(f"cache hit for {fname}")
        except FileNotFoundError:
            print(f"cache miss for {fname}")
            truncate_to_frame = int(truncate_data_to_time_s/self.frametime_s)
            data_trunc = self.data[:min(len(self.data), truncate_to_frame)]
            trig_inds = npyfilter.fasttrig_filter_trigger(data_trunc, trigger_filter, 
                                                            trigger_threshold)
            np.save(fname, trig_inds)        
        return trig_inds
        
    def trigger_pulses_or_load_from_cache(self, trigger_filter, trigger_threshold, truncate_data_to_time_s):
        trig_ind = self._trigger_pulses_or_load_from_cache(trigger_filter=trigger_filter, 
                                                            trigger_threshold=trigger_threshold, 
                                                            truncate_data_to_time_s=truncate_data_to_time_s) 
        trig_rel_time_s = trig_ind*self.frametime_s
        time_since_last_s = np.diff(trig_rel_time_s, prepend=trig_rel_time_s[0])
        time_to_next_s = np.diff(trig_rel_time_s, append=trig_rel_time_s[-1])
        self.df = self.df.with_columns(trig_ind=trig_ind, 
                                       time_since_last_s=time_since_last_s, 
                                       time_to_next_s=time_to_next_s)

    def calculate_noise_triggers(self, noise_n_samples, n_dead_samples_after_previous_pulse, max_triggers=1000):
        trig_ind = self.df["trig_ind"]
        noise_trig_ind = npyfilter.get_noise_trigger_inds(trig_ind, n_dead_samples_after_previous_pulse, 
                           n_record_samples=noise_n_samples, max_noise_triggers=1000)
        return NpyNoise(self.fname_npy, self.data, noise_trig_ind, self.frametime_s, noise_n_samples)
    
    def calculate_spikeyness_pretrig_mean_pulse_rms(self, frontload_n_samples):
        spikeyness, pretrig_mean, pulse_rms = npyfilter.calculate_spikeyness_pretrig_mean_pulse_rms(self.data, 
                            frontload_n_samples, self.npre, self.nsamples, 
                            self.pulse_polarity, self.df["trig_ind"])
        self.df = self.df.with_columns(spikeyness=spikeyness,
                                       pretrig_mean=pretrig_mean,
                                       pulse_rms=pulse_rms)
        
    def title(self):
        return self.fname_npy
        
    def spikeyness_debug_plot(self, min_time_since_last_s, min_time_to_next_s, 
                              spikeyness_threshold_non_foil_only,
                              spikeyness_threshold_foil_plus_non_foil):
        isolated_bool = (self.df["time_since_last_s"]>min_time_since_last_s)&(self.df["time_to_next_s"]>min_time_to_next_s)
        spikeyness_of_isolated = self.df.filter(isolated_bool)["spikeyness"].to_numpy()
        plt.figure()
        plt.hist(spikeyness_of_isolated, np.linspace(0,20*np.median(spikeyness_of_isolated),100))
        plt.xlabel("spikeyness")
        plt.ylabel("number of occurences")
        plt.axvline(spikeyness_threshold_non_foil_only, label=f"{spikeyness_threshold_non_foil_only=:7.2g}",color="r")
        plt.axvline(spikeyness_threshold_foil_plus_non_foil, label=f"{spikeyness_threshold_foil_plus_non_foil=:7.2g}",color="k")
        plt.title(f"""{self.title()}
                  {min_time_since_last_s=:5.2g} {min_time_to_next_s=:5.2g}""")
        plt.legend()
        plt.yscale("log")

    def calculate_average_pulse(self, min_time_since_last_s, min_time_to_next_s, spikeyness_threshold_foil_plus_non_foil):
        filtered_rows = self.df.filter(pl.col("time_since_last_s")>min_time_since_last_s,
                                      pl.col("time_to_next_s")>min_time_to_next_s,
                                      pl.col("spikeyness")<spikeyness_threshold_foil_plus_non_foil)
        trig_inds = filtered_rows["trig_ind"].to_numpy()
        self.average_pulse_obj = NpyAveragePulse("average pulse", trig_inds, self.data, self.npre, self.nsamples)
        return self.average_pulse_obj
    
    def calculate_filter(self, avg_pulse_values, noise_autocorr, noise_psd,
                         filter_pretrigger_ignore_samples, 
                         filter_orthogonal_to_exponential_time_constant_ms, filter_choice="noconst"):
        filter_obj = mass.ExperimentalFilter(avg_pulse_values, self.npre-filter_pretrigger_ignore_samples,
                                 noise_psd, sample_time=self.frametime_s, 
                                 noise_autocorr=noise_autocorr,
                                 tau=filter_orthogonal_to_exponential_time_constant_ms)
        filter_obj.compute()
        print("predicted resolutions")
        filter_obj.report(std_energy=am241_Q_eV)
        self._filter_obj = filter_obj
        chosen_filter = getattr(filter_obj, f"filt_{filter_choice}")
        try:
            self.chosen_filter_v_dv = filter_obj.v_dv[filter_choice]
        except:
            self.chosen_filter_v_dv = np.nan
        self.chosen_filter = chosen_filter
        self.filter_choice = filter_choice
        return chosen_filter
    
    def filter(self):
        trig_inds = self.df["trig_ind"]
        filt_value, residual_rms, filt_value_template = npyfilter.filter_and_residual_rms(self.data, 
                self.chosen_filter, self.average_pulse_obj.values(), trig_inds, self.npre, self.nsamples,
                self.pulse_polarity)
        self.df = self.df.with_columns(filt_value=filt_value, residual_rms=residual_rms, 
                                       filt_value_template=filt_value_template)
        
    def classify(self, min_time_since_last_s, min_time_to_next_s, spikeyness_threshold_non_foil_only,
                 spikeyness_threshold_foil_plus_non_foil, max_residual_rms):
        classification_meaning = {0: "foil clean",
                          1: "foil + non_foil",
                          2: "non_foil_only",
                          3: "last too close",
                          4: "next too close",
                          5: "last and next too close",
                          6: "high residual_rms"}
        c = np.zeros(len(self.df), dtype=int)
        time_since_last_s = self.df["time_since_last_s"].to_numpy()
        time_to_next_s = self.df["time_to_next_s"].to_numpy()
        spikeyness = self.df["spikeyness"].to_numpy()
        residual_rms = self.df["residual_rms"].to_numpy()
        c[(c==0)&(time_since_last_s<min_time_since_last_s)&(time_to_next_s<min_time_to_next_s)]=5
        c[(c==0)&(time_since_last_s<min_time_since_last_s)]=3
        c[(c==0)&(spikeyness>spikeyness_threshold_non_foil_only)]=2
        c[(c==0)&(time_to_next_s<min_time_to_next_s)]=4
        c[(c==0)&(spikeyness>spikeyness_threshold_foil_plus_non_foil)]=1
        c[(c==0)&(residual_rms>max_residual_rms)]=6
        self.df = self.df.with_columns(classification=c)
        return classification_meaning
    
    def classification_debug_plots(self, class_meaning):
        classification = self.df["classification"].to_numpy()
        trig_inds = self.df["trig_ind"].to_numpy()
        for c, c_meaning in class_meaning.items():
            inds = np.nonzero(classification==c)[0]
            c_meaning = class_meaning[c]
            print(f"{c=} {c_meaning} {len(inds)=}")
            npyfilter.plot_inds(self.data, self.npre, self.nsamples, trig_inds[inds], label=f"{c_meaning}: {len(inds)} pulses of type {c} (up to 50 shown)", max_pulses_to_plot=50)
    
    def classification_debug_plots_ax(self, class_meaning):
        classification = self.df["classification"].to_numpy()
        trig_inds = self.df["trig_ind"].to_numpy()
        fig, axs = plt.subplots(3,3)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        for i, (c, c_meaning) in enumerate(class_meaning.items()):
            inds = np.nonzero(classification==c)[0]
            c_meaning = class_meaning[c]
            print(f"{c=} {c_meaning} {len(inds)=}")
            plt.sca(axs.flat[i])
            npyfilter.plot_inds(self.data, self.npre, self.nsamples, trig_inds[inds], 
            label=f"{c_meaning}: {len(inds)} pulses of type {c} (up to 50 shown)", 
            max_pulses_to_plot=50, newfig=False)


    def filt_value_to_energy_with_ptmean_correlation_removal(self, median_energy):
        fv_subset, pretrig_mean_subset = self.df.filter(pl.col("classification")==0)[["filt_value", "pretrig_mean"]]
        filt_value, pretrig_mean = self.df[["filt_value", "pretrig_mean"]]
        median_vs_subset = np.median(fv_subset)
        energy_uncorrected = filt_value*median_energy/median_vs_subset

        slope, info = mass.core.analysis_algorithms.drift_correct(
            pretrig_mean_subset, fv_subset)
        median_pt_mean = info['median_pretrig_mean']
        fv_corrected =  filt_value* (1+(pretrig_mean-median_pt_mean)*slope)
        energy_corrected = fv_corrected*median_energy/median_vs_subset
        self.df = self.df.with_columns(energy_uncorrected=energy_uncorrected, energy=energy_corrected)

class BinAnalyzer(NpyAnalyzer):
    def __init__(self, fname_bin, pulse_polarity, npre, nsamples):
        self.fname_npy = fname_bin # this more likely to just work elsehwere
        self.data = np.memmap(fname_bin, dtype=np.int16, mode='r', offset=68)
        self.header = self.read_header(fname_bin)
        self.pulse_polarity=pulse_polarity
        self.npre=npre
        self.nsamples=nsamples
        self.df = pl.DataFrame()
        self.frametime_s=1/self.header["sample_rate_hz"][0]

    def read_header(self, fname_bin):
        header_dtype = np.dtype([("format", np.uint32), ("schema", np.uint32), ("sample_rate_hz", np.float64), ("data reduction factor", np.int16), ("voltage scale", np.float64), ("aquisition flags", np.uint16), ("start_time", np.uint64, 2), ("stop_time", np.uint64, 2), ("number of samples", np.uint64)])
        header_np = np.memmap(fname_bin, dtype=header_dtype, mode='r', offset=0, shape=1)
        header = pl.from_numpy(header_np)
        return header
        

# analyzer = truebq_analysis.TrueBqNpyAnalyzer(inputs)
# analyzer.trigger_pulses_or_load_from_cache(trigger_inputs)
# noise_info = analyzer.calculate_noise_triggers(noise_trigger_inputs)
# noise_info.plot_noise()
# noise_info_long = analyzer.calculate_noise_triggers(noise_trigger_inputs_long)
# noise_info_long.plot()
# analyzer.calculate_spikeyness(per_pulse_quantity_inputs)
# analyzer.classification_debug_plots()
# analyzer.calculate_average_pulse()
# analyzer.calculate_filter()

