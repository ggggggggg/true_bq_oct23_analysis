import marimo

__generated_with = "0.7.11"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import pylab as plt
    import polars as pl
    return mo, np, pl, plt


@app.cell
def __():
    bin_path = r"C:\Users\radgroup\src\data\240722_110250_GDES001 2.7 V\Dev2_ai3\data.bin"
    return bin_path,


@app.cell
def __(bin_path, np):
    data = np.memmap(bin_path, dtype=np.int16, mode='r', offset=68)
    return data,


@app.cell
def __(bin_path, np, pl):
    header_dtype = np.dtype([("format", np.uint32), ("schema", np.uint32), ("sample_rate_hz", np.float64), ("data reduction factor", np.int16), ("voltage scale", np.float64), ("aquisition flags", np.uint16), ("start_time", np.uint64, 2), ("stop_time", np.uint64, 2), ("number of samples", np.uint64)])
    header_np = np.memmap(bin_path, dtype=header_dtype, mode='r', offset=0, shape=1)
    sample_rate_hz = header_np["sample_rate_hz"]
    header = pl.from_numpy(header_np)
    header
    return header, header_dtype, header_np, sample_rate_hz


@app.cell
def __(data, mo, plt):
    plt.plot(data[:100000:10])
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
