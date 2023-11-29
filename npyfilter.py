from numba import njit
import numba.typed
import numpy as np
import scipy
import pylab as plt

@njit
def fasttrig_filter_trigger(data, filter_in, threshold):
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
    prog_step = jmax//100
    prog_ticks = 0
    while j <= jmax:
        if j%prog_step==0:
            prog_ticks+=1
            print(f"fasttrig_filter_trigger {prog_ticks}/{100}")
        a,b = b,c
        cache[:] = data[j:(j+filter_len)]
        c = np.dot(cache, filter)
        if b>threshold and b>=c and b>a:
            inds.append(j)
        j+=1
    return np.array(inds)

@njit 
def fast_apply_filter(data, filter):
    filter_len = len(filter)
    filter_out = np.zeros(len(data)-len(filter))
    j = 0
    jmax = len(data)-filter_len-1
    while j <= jmax:
        filter_out[j] = np.dot(data[j:(j+filter_len)], filter)
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

def plot_inds(data, npre, nsamples, inds, label, max_pulses_to_plot=40):
    import pylab as plt
    cmap = plt.matplotlib.colormaps.get_cmap("rainbow")
    plt.figure()
    for j, i in enumerate(inds):
        if j >= max_pulses_to_plot:
            break
        color = cmap(j/min(len(inds), max_pulses_to_plot))
        trace = data[i-npre:i+nsamples-npre]
        plt.plot(trace, "--", color=color, label=f"{i}", picker=True, pickradius=5)
        # if plot_modeled:
        #     plt.plot(dsoff.offFile.modeledPulse(i), color=color)       
    plt.legend()
    plt.title(f"{label}")
    plt.xlabel("sample number")
    plt.ylabel("signal (arb)")
    plt.gcf().canvas.mpl_connect('pick_event', onpick)

def spectrum_from_pulse(noise_pulses, frametime_s):
    """noise_pulses[:,0] is the first pulse"""
    nsamples = noise_pulses.shape[0]
    spectrum = mass.mathstat.power_spectrum.PowerSpectrum(nsamples // 2, dt=frametime_s)
    window = np.ones(nsamples)
    for i in range(noise_pulses.shape[1]):
        spectrum.addDataSegment(noise_pulses[:,i], window=window)
    return spectrum
