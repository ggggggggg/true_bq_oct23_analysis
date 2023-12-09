from numba import njit
import numba.typed
import numpy as np
import scipy
import pylab as plt
import mass

def calculate_spikeyness_pretrig_mean_pulse_rms(data, frontload_n_samples, npre, nsamples, polarity, trig_inds):
        spikeyness = np.zeros(len(trig_inds))
        pretrig_mean = np.zeros(len(trig_inds))
        pulse_rms = np.zeros(len(trig_inds))
        for i, trig_ind in enumerate(trig_inds):
            pulse = data[trig_ind-npre:trig_ind+nsamples-npre]*polarity
            pretrig_mean[i] = np.mean(pulse[:npre//2])
            peakval = np.amax(pulse)-pretrig_mean[i]
            pulse_rms[i] = np.sqrt(np.sum((pulse[npre:]-pretrig_mean[i])**2))
            # max_deriv = np.amax(np.diff(pulse))
            max_neg_deriv = -np.amin(np.diff(pulse[::10]))/10
            pulse_area = np.sum(pulse[npre:]-pretrig_mean[i])
            spikeyness[i] = max_neg_deriv/peakval
            frontload_pulse_area = np.sum(pulse[npre:npre+frontload_n_samples]-pretrig_mean[i])
            backload_pulse_area = np.sum(pulse[npre+frontload_n_samples:]-pretrig_mean[i])
            spikeyness[i] = max_neg_deriv/peakval*frontload_pulse_area/pulse_area
            # spikeyness[i] = frontload_pulse_area/backload_pulse_area
            pretrig_mean[i] *= polarity
        return spikeyness, pretrig_mean, pulse_rms

def filter_and_residual_rms(data, chosen_filter, avg_pulse, trig_inds, npre, nsamples, polarity):
    filt_value = np.zeros(len(trig_inds))
    residual_rms = np.zeros(len(trig_inds))
    filt_value_template = np.zeros(len(trig_inds))
    template = avg_pulse-np.mean(avg_pulse)
    template = template/np.sqrt(np.dot(template, template))
    for i in range(len(trig_inds)):
        j = trig_inds[i]
        pulse = data[j-npre:j+nsamples-npre]*polarity
        pulse = pulse - pulse.mean()
        filt_value[i] = np.dot(chosen_filter, pulse)
        filt_value_template[i] = np.dot(template, pulse)
        residual = pulse-template*filt_value_template[i]
        residual_std_dev = np.std(residual)
        residual_rms[i] = residual_std_dev  
    return filt_value, residual_rms, filt_value_template

def gather_pulse_from_inds(data, npre, nsamples, inds):
    pulses = np.zeros((nsamples, len(inds)))
    for i, ind in enumerate(inds):
        pulses[:,i] = data[ind-npre:ind+nsamples-npre]
    return pulses

@njit
def fasttrig_filter_trigger(data, filter_in, threshold):
    assert threshold>0, "algorithm assumes we trigger with positiv threshold, change sign of filter_in to accomodate"
    filter_len = len(filter_in)
    inds = []
    jmax = len(data)-filter_len-1
    # njit only likes float64s, so I'm trying to force float64 use without allocating a ton of memory
    cache = np.zeros(len(filter_in))
    filter = np.zeros(len(filter_in))
    filter[:]=filter_in
    # intitalize a,b,c
    j=0
    cache[:] = data[j:(j+filter_len)]
    b = np.dot(cache, filter)
    a=b # won't be used, just need same type
    j=1
    cache[:] = data[j:(j+filter_len)]
    c = np.dot(cache, filter)
    j=2
    ready = False
    prog_step = jmax//100
    prog_ticks = 0
    while j <= jmax:
        if j%prog_step==0:
            prog_ticks+=1
            print(f"fasttrig_filter_trigger {prog_ticks}/{100}")
        a,b = b,c
        cache[:] = data[j:(j+filter_len)]
        c = np.dot(cache, filter)
        if b>threshold and b>=c and b>a and ready:
            inds.append(j)
            ready = False
        if b<0: # hold off on retriggering until we see opposite sign slope
            ready=True
        j+=1
    return np.array(inds)


@njit 
def fast_apply_filter(data, filter_in):
    cache = np.zeros(len(filter_in))
    filter = np.zeros(len(filter_in))
    filter_len = len(filter)
    filter_out = np.zeros(len(data)-len(filter))
    j = 0
    jmax = len(data)-filter_len-1
    while j <= jmax:
        cache[:] = data[j:(j+filter_len)]
        filter_out[j] = np.dot(cache, filter)
    return filter_out

def get_noise_trigger_inds(pulse_trigger_inds, n_dead_samples_after_previous_pulse, 
                           n_record_samples, max_noise_triggers):
    diffs = np.diff(pulse_trigger_inds)
    inds = []
    for i in range(len(diffs)):
        if diffs[i] > n_dead_samples_after_previous_pulse:
            n_make = (diffs[i]-n_dead_samples_after_previous_pulse)//n_record_samples
            ind0 = pulse_trigger_inds[i]+n_dead_samples_after_previous_pulse
            for j in range(n_make):
                inds.append(ind0+n_record_samples*j)
                if len(inds) == max_noise_triggers:
                    return np.array(inds)
    return np.array(inds)


event_g = None
def onpick(event):
    global event_g
    event_g = event
    print(event_g.artist.get_label())

def plot_inds(data, npre, nsamples, inds, label, max_pulses_to_plot=40, newfig=True):
    import pylab as plt
    cmap = plt.matplotlib.colormaps.get_cmap("rainbow")
    if newfig:
        plt.figure()
    for j, i in enumerate(inds):
        if j >= max_pulses_to_plot:
            break
        color = cmap(j/min(len(inds), max_pulses_to_plot))
        trace = data[i-npre:i+nsamples-npre]
        plt.plot(trace, "--", color=color, label=f"{i}", picker=True, pickradius=5)
        # if plot_modeled:
        #     plt.plot(dsoff.offFile.modeledPulse(i), color=color)       
    if len(inds)>0:
        plt.legend()
    plt.title(f"{label}")
    plt.xlabel("sample number")
    plt.ylabel("signal (arb)")
    plt.gcf().canvas.mpl_connect('pick_event', onpick)

def spectrum_from_pulse(noise_pulses, frametime_s):
    """noise_pulses[:,0] is the first pulse"""
    nsamples, npulses = noise_pulses.shape
    spectrum = mass.mathstat.power_spectrum.PowerSpectrum(nsamples // 2, dt=frametime_s)
    window = np.ones(nsamples)
    for i in range(npulses):
        spectrum.addDataSegment(noise_pulses[:,i], window=window)
    return spectrum

def median_absolute_deviation(x):
    med = np.median(x)
    return np.median(np.abs(x-med))
def mad_threshold(x, n_sigma=5):
    med = np.median(x)
    mad = median_absolute_deviation(x)
    sigma = mad*1.4826
    return med+5*sigma