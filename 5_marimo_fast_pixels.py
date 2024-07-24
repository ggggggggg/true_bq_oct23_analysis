import marimo

__generated_with = "0.7.11"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        r"""
        # warning you must run marimo edit from
        C:\Users\radgroup\src\true_bq_oct23_analysis
        # for fast cspec06/GDES001 pixels
        you can change fname_npy and all the spikeyness threshold settings to make other pixels work too
        """
    )
    return


@app.cell
def __():
    import os
    import numpy as np
    import pylab as plt
    import polars as pl
    import pickle
    import npyfilter
    import scipy
    import truebq_analysis
    import mass
    import truebqlines
    import marimo as mo
    return (
        mass,
        mo,
        np,
        npyfilter,
        os,
        pickle,
        pl,
        plt,
        scipy,
        truebq_analysis,
        truebqlines,
    )


@app.cell
def __(np):
    #trigger params (cached)
    fname_npy = r"C:\Users\radgroup\src\data\240722_110250_GDES001 2.7 V\Dev2_ai3\data.bin"
    polarity = -1 # postive for positive oging pulses, negative for negative going
    trigger_filter = np.array([-1]*20+[+1]*20,dtype=float) * polarity # negate for negative pulses
    trigger_threshold = 200
    truncate_data_to_time_s = 1e9 # for faster triggering use a smaller value
    return (
        fname_npy,
        polarity,
        trigger_filter,
        trigger_threshold,
        truncate_data_to_time_s,
    )


@app.cell
def __():
    # noise params
    noise_n_dead_samples_after_previous_pulse = 70000
    noise_long_n_samples = 100000
    # for full analysis use a ridiculously large value like 1e9
    # script will auto re-use triggering if all other parameters are identical, so you only need
    # to wait once
    return noise_long_n_samples, noise_n_dead_samples_after_previous_pulse


@app.cell
def __():
    # inputs
    # pulse selection quantities
    min_time_since_last_s = 0.2
    min_time_to_next_s = 0.010
    min_time_since_last_s_for_average_pulse = 0.6

    # filter inputs
    nsamples = 2000
    npre = 1000
    filter_orthogonal_to_exponential_time_constant_ms = 4
    filter_pretrigger_ignore_samples = 10 # this value to exclude rising samples from pretrigger period

    # pulse classification inputs
    max_residual_rms = 60
    spikeyness_threshold_foil_plus_non_foil = 0.003
    spikeyness_threshold_non_foil_only = 0.006
    frontload_n_samples = 100

    # energy calibration inputs
    am241_Q_eV = 5637.82e3

    # hist inputs
    roi_lo = 5.4e6
    roi_hi = 6.0e6
    return (
        am241_Q_eV,
        filter_orthogonal_to_exponential_time_constant_ms,
        filter_pretrigger_ignore_samples,
        frontload_n_samples,
        max_residual_rms,
        min_time_since_last_s,
        min_time_since_last_s_for_average_pulse,
        min_time_to_next_s,
        npre,
        nsamples,
        roi_hi,
        roi_lo,
        spikeyness_threshold_foil_plus_non_foil,
        spikeyness_threshold_non_foil_only,
    )


@app.cell
def __(
    fname_npy,
    noise_n_dead_samples_after_previous_pulse,
    npre,
    nsamples,
    plt,
    polarity,
    trigger_filter,
    trigger_threshold,
    truebq_analysis,
    truncate_data_to_time_s,
):
    a=None # give order to cells since analyzer is mutable
    analyzer = truebq_analysis.BinAnalyzer(fname_npy, polarity, npre, nsamples)
    analyzer.trigger_pulses_or_load_from_cache(trigger_threshold=trigger_threshold, 
                                               trigger_filter=trigger_filter,
                                               truncate_data_to_time_s=truncate_data_to_time_s)
    noise = analyzer.calculate_noise_triggers(nsamples, noise_n_dead_samples_after_previous_pulse)
    noise.plot(f"noise record length {nsamples=}")
    plt.tight_layout()
    a=4 # just to give order to cells since analyzer is mutable
    # mo.mpl.interactive(plt.gcf()) # this plot isnt super useful, lets comment it out
    return a, analyzer, noise


@app.cell
def __(
    a,
    analyzer,
    mo,
    noise_long_n_samples,
    noise_n_dead_samples_after_previous_pulse,
    plt,
):
    b=a # give order to cells since analyzer is mutable
    noise_long = analyzer.calculate_noise_triggers(noise_long_n_samples, noise_n_dead_samples_after_previous_pulse, max_triggers=200)
    noise_long.plot(f"noise long {noise_long_n_samples=}")
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return b, noise_long


@app.cell
def __(analyzer, b):
    b # give order to cells since analyzer is mutable
    analyzer.df
    return


@app.cell
def __(
    analyzer,
    b,
    frontload_n_samples,
    min_time_since_last_s,
    min_time_to_next_s,
    mo,
    plt,
    spikeyness_threshold_foil_plus_non_foil,
    spikeyness_threshold_non_foil_only,
):
    c=b # give order to cells since analyzer is mutable
    analyzer.calculate_spikeyness_pretrig_mean_pulse_rms(frontload_n_samples)
    analyzer.spikeyness_debug_plot(min_time_since_last_s=min_time_since_last_s,
                                   min_time_to_next_s=min_time_to_next_s,
                                   spikeyness_threshold_foil_plus_non_foil=spikeyness_threshold_foil_plus_non_foil,
                                   spikeyness_threshold_non_foil_only=spikeyness_threshold_non_foil_only)
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return c,


@app.cell
def __(
    analyzer,
    c,
    min_time_since_last_s_for_average_pulse,
    min_time_to_next_s,
    mo,
    plt,
    spikeyness_threshold_foil_plus_non_foil,
):
    d=c # give order to cells since analyzer is mutable
    avg_pulse_obj = analyzer.calculate_average_pulse(min_time_since_last_s_for_average_pulse, min_time_to_next_s, 
                                                     spikeyness_threshold_foil_plus_non_foil)
    avg_pulse_obj.plot()
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return avg_pulse_obj, d


@app.cell
def __(
    analyzer,
    avg_pulse_obj,
    d,
    filter_orthogonal_to_exponential_time_constant_ms,
    filter_pretrigger_ignore_samples,
    noise,
):
    e=d # give order to cells since analyzer is mutable
    analyzer.calculate_filter(avg_pulse_obj.values(), noise.autocorr(), noise.spectrum.spectrum(), 
                              filter_pretrigger_ignore_samples,
                              filter_orthogonal_to_exponential_time_constant_ms)
    analyzer.filter()
    return e,


@app.cell
def __(
    am241_Q_eV,
    analyzer,
    e,
    max_residual_rms,
    min_time_since_last_s,
    min_time_to_next_s,
    mo,
    plt,
    spikeyness_threshold_foil_plus_non_foil,
    spikeyness_threshold_non_foil_only,
):
    f=e # give order to cells since analyzer is mutable
    # max_residual_rms = npyfilter.mad_threshold(analyzer.df["residual_rms"].to_numpy(), n_sigma=residual_rms_n_sigma)
    class_meaning = analyzer.classify(min_time_since_last_s=min_time_since_last_s, 
                                      min_time_to_next_s=min_time_to_next_s,
                                      spikeyness_threshold_foil_plus_non_foil=spikeyness_threshold_foil_plus_non_foil,
                                      spikeyness_threshold_non_foil_only=spikeyness_threshold_non_foil_only,
                                      max_residual_rms=max_residual_rms)
    analyzer.classification_debug_plots_ax(class_meaning)
    analyzer.filt_value_to_energy_with_ptmean_correlation_removal(median_energy=am241_Q_eV)
    analyzer.df.write_parquet("out.parquet")
    mo.mpl.interactive(plt.gcf())
    return class_meaning, f


@app.cell
def __(
    analyzer,
    class_meaning,
    f,
    min_time_since_last_s,
    mo,
    np,
    pl,
    plt,
    roi_hi,
    roi_lo,
):
    f # give order to cells since analyzer is mutable

    def makeplot():
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
        return df, bin_edges
    df, bin_edges = makeplot()
    mo.mpl.interactive(plt.gcf())
    return bin_edges, df, makeplot


@app.cell
def __(am241_Q_eV, f, mass, np, npre, npyfilter, nsamples, pl, plt):
    f # give order to cells
    def binsize(x):
        return x[1]-x[0]
    def midpoints(x):
        return (x[1:]+x[:-1])/2
    def plot_hist_mark_roi(data, df, bin_edges, roi_lo, roi_hi):
        plt.figure(figsize=(16,6))
        plt.subplot(121)
        counts, _ = np.histogram(df["energy_classified"].to_numpy(), bin_edges)
        roi_plot_inds = (midpoints(bin_edges) > roi_lo) & (midpoints(bin_edges)<roi_hi)
        plt.plot(midpoints(bin_edges), counts, drawstyle="steps-mid")
        plt.plot(midpoints(bin_edges)[roi_plot_inds], counts[roi_plot_inds], drawstyle="steps-mid", color="r",
                         label="plotted pulses ROI")
        plt.legend()
        plt.yscale("log")
        plot_inds = df.filter(pl.col("energy_classified")>roi_lo, pl.col("energy_classified")<roi_hi)["trig_ind"]
        plt.subplot(122)
        label=""
        npyfilter.plot_inds(data, npre, nsamples, plot_inds, label, max_pulses_to_plot=40, newfig=False)
        plt.gca().get_legend().remove()

    def plot_hist_with_fit(analyzer, df, bin_edges=np.arange(-100e3,100e3,1e3)+am241_Q_eV):
        counts, _ = np.histogram(df["energy_classified"].to_numpy(), bin_edges)
        model = mass.get_model("Am241Q", has_tails=True)
        result = model.fit(bin_centers=midpoints(bin_edges), data=counts)
        result.plotm()
        filter_choice = analyzer.filter_choice
        chosen_filter_v_dv = analyzer.chosen_filter_v_dv
        chosen_filter_predicted_fwhm = am241_Q_eV/chosen_filter_v_dv
        plt.title(f"{plt.gca().get_title()}\n{filter_choice=} {chosen_filter_predicted_fwhm=:.2f}")
    return binsize, midpoints, plot_hist_mark_roi, plot_hist_with_fit


@app.cell
def __(analyzer, df, f, mo, plot_hist_with_fit, plt):
    f # give order to cells
    result = plot_hist_with_fit(analyzer, df)
    mo.mpl.interactive(plt.gcf())
    return result,


@app.cell
def __(analyzer, bin_edges, df, f, mo, plot_hist_mark_roi, plt, roi_lo):
    f
    plot_hist_mark_roi(analyzer.data, df, bin_edges, roi_lo=0, roi_hi=roi_lo)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
