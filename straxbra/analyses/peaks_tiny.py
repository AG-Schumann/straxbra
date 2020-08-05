import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from . import analysis_tools as at
import straxbra


@straxbra.tiny_analysis(requires=('peaks', 'peak_classification'))
def plot_peaks(
            peaks,
            time_range=None,
            t_reference=None,
            peak_idx=None,
            s0_color='gray',
            s1_color='b',
            s2_color='g',
            title='',
            xlabel='t [us]',
            ylabel="Intensity [PE/ns]",
            figsize=(10, 4),
            is_log=False,
            **kwargs):
    """Plots peak waveforms in given selection.

    Arguments for decorator:
    context -- straxbra context
    run_id  -- 5-digit zero padded run-id (str)

    Arguments:
    peaks   -- Provided by decorator (ndarray)

    Keyword arguments:
    time_range -- Time range (ns: Unix timestamp) in that peaks
                  are plotted (default None, tuple)
    t_reference-- Reference time. Starttime of plotting (ns, Unix timestamp) If None
                  start time of first peak in selection is used (default None, int)
    peak_idx   -- Index / Indecies or boolean array to select peaks to plot
                  (default None, int / list of ints / bool list with len of peaks)
    s0_color   -- Plot color of peaks (default 'gray')
    s1_color   -- Plot color of main S1 (default 'b')
    s2_color   -- Plot color of main S2 (default 'g')
    title      -- Title of Plot   (str)
    xlabel     -- Label of x-axis (str)
    ylabel     -- Label of y-axis (str)
    figsize    -- mpl figsize (default (10,4), tuple)
    is_log     -- Whether or not to plot y-axis logarithmicly (default False)
    kwargs     -- Any kwargs plt.plot accepts, except color.
    """

    if time_range is None and peak_idx is None:
        raise RuntimeError('Kwarg missing. Give either time_range or peak_idx.')
    elif time_range is not None and peak_idx is not None:
        raise ValueError('Expected time_range OR peak_idx kwarg, got both.')

    if 'color' in kwargs or 'c' in kwargs:
        raise ValueError('Give plot color via s0_color, s1_color and s2_color not color or c.')

    if time_range is not None:
        endtime = peaks['time'] + peaks['length'] * peaks['dt']
        peaks = peaks[(peaks['time'] >= time_range[0]) &
                      (endtime <= time_range[1])]
    else:
        peaks = peaks[peak_idx]

    if t_reference is None:
        t_reference = peaks[0]['time']

    plt.figure(figsize=figsize)
    plt.axhline(0, c='k', alpha=0.2)

    for p in peaks:
        at.plot_peak(p,
                    t0=t_reference,
                    color={0: s0_color, 1: s1_color, 2: s2_color}[p['type']],
                    **kwargs)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if is_log:
        plt.ylim(0.1, None)
        plt.yscale('log')
    else:
        plt.axhline(0, c='k', alpha=0.2)

    plt.tight_layout()



@straxbra.tiny_analysis(requires=('peaks', 'peak_basics', 'peak_classification'))
def peak2dhist(
            peaks,
            x='area',
            y='width',
            x_quantile=5,
            y_quantile=5,
            cuts=None,
            xbins=None,
            ybins=None,
            xlabel=None,
            ylabel=None,
            title='',
            cmap='plasma',
            xlog=True,
            ylog=True,
            figsize=(9, 5),
            **kwargs):

    """Show 2D histogram of peaks versus area(x) and width(y) (default).

    Arguments for decorator:
    context -- straxbra context
    run_id  -- 5-digit zero padded run-id (str)

    Arguments:
    peaks   -- Provided by decorator (ndarray)

    Keyword arguments:
    x          -- Field name in peaks that is plotted on x-axis
    y          -- Field name in peaks that is plotted on y-axis
    x_quantile -- Selection index of x-field if  given field is multi-dim
    y_quantile -- Selection index of y-field if  given field is multi-dim
    cuts       -- Boolean array or list of indices to apply cuts to peaks
                  if None nothing is applied (default None)
    xbins      -- Number of bins or list of bin-edges (see np.histogram2d)
                  If None default logspace is used (default None)
    ybins      -- Number of bins or list of bin-edges (see np.histogram2d)
                  If None default logspace is used (default None)
    xlabel     -- Label of x-axis. If None use x (default None, str)
    ylabel     -- Label of y-axis. If None use y (default None, str)
    title      -- Title of Plot   (str)
    cmap       -- Mpl-colormap used (default plasma)
    xlog       -- Whether or not to plot x-axis logarithmicly (default True)
    ylog       -- Whether or not to plot y-axis logarithmicly (default True)
    figsize    -- mpl figsize (default (9, 5), tuple)
    kwargs     -- Any kwargs plt.pcolormesh accepts
    """

    if not cuts is None:
        peaks = peaks[cuts]

    xvals = peaks[x] if not isinstance(peaks[0][x], np.ndarray) else peaks[x][:, x_quantile]
    yvals = peaks[y] if not isinstance(peaks[0][y], np.ndarray) else peaks[y][:, y_quantile]

    xbins = np.logspace(0, 6, 100) if xbins is None else xbins
    ybins = np.logspace(0, 5, 100) if ybins is None else ybins

    counts, _, _ = np.histogram2d(xvals, yvals, bins=[xbins, ybins])

    plt.figure(figsize=figsize)
    im = plt.pcolormesh(xbins, ybins, counts.T, cmap=cmap, norm=colors.LogNorm(), **kwargs)

    plt.title(title)
    plt.xlabel(xlabel if xlabel else x)
    plt.ylabel(ylabel if ylabel else y)

    if xlog: plt.xscale('log')
    if ylog: plt.yscale('log')
    plt.colorbar()
    plt.tight_layout()



@straxbra.tiny_analysis(requires=('peaks', ))
def pmt_activity(
            peaks,
            cuts=None,
            bins_per_sec=1,
            thresh_pos=0,
            thresh_neg=0,
            log_color=True,
            xlabel='time since run-start [s]',
            ylabel='PMT Nr.',
            clabel=None,
            title='',
            cmap='plasma',
            figsize=(20, 10),
            **kwargs):
    """2dHist of how active each PMT is.
       For each PMT it's checked to how many peaks per unit time (default 1 sec)
       are above or below an area threshold value (default "> 0 PE"(+) and "< 0 PE"(-)).
       Both counts are represented in the color-scale.

    Arguments for decorator:
    context -- straxbra context
    run_id  -- 5-digit zero padded run-id (str)

    Arguments:
    peaks   -- Provided by decorator (ndarray)

    Keyword arguments:
    cuts         -- Boolean array or list of indices to apply cuts to peaks
                    if None nothing is applied (default None)
    bins_per_sec -- Granularity of time-bins
    thresh_pos   -- Threshold value in area [PE] above which a pos-area
                    contribution is counted (default 0; float,int)
    thresh_neg   -- Threshold value in area [PE] below which a neg-area
                    contribution is counted (default 0; float,int)
    log_color    -- Whether or not to have a logarithmic color scale
    xlabel       -- Label of x-axis (str)
    ylabel       -- Label of y-axis (str)
    clabel       -- Label of colorbar. If None default is used (default None, str)
    title        -- Title of Plot (default '', str)
    cmap         -- Colormap of colormesh (default 'plasma', see mpl docs)
    figsize      -- mpl figsize (default (20, 10), tuple)
    kwargs       -- Any kwargs plt.pcolormesh accepts
    """
    if cuts is not None:
        peaks = peaks[cuts]

    n_pmts = len(peaks['area_per_channel'][0])
    p_time = (peaks['time'] - peaks['time'][0])/1e9

    t_bin_e = np.linspace(0, np.ceil(max(p_time)), int(np.ceil(max(p_time)*bins_per_sec))+bins_per_sec)
    t_bin_c = (t_bin_e[1:] + t_bin_e[:-1]) / 2

    pmt_bin_e = np.linspace(-0.5, 3*(n_pmts-1)+1.5, 3*n_pmts)

    pmts = np.full((n_pmts*3-1, len(t_bin_c)), np.nan)

    plt.figure(figsize=figsize)
    for pmt in range(n_pmts):
        pos = peaks['area_per_channel'][:, pmt] > thresh_pos
        neg = peaks['area_per_channel'][:, pmt] < thresh_neg

        pmts[pmt*3+1], _ = np.histogram(p_time[pos], bins=t_bin_e)
        pmts[pmt*3]  , _ = np.histogram(p_time[neg], bins=t_bin_e)

    norm = colors.LogNorm() if log_color else None
    kwargs.pop('norm', '')
    plt.pcolormesh(t_bin_e, pmt_bin_e, pmts, cmap=cmap, norm=norm, **kwargs)

    # colorbar
    cbar = plt.colorbar()
    sec_label = "" if bins_per_sec == 1 else f"{1/bins_per_sec:0.1}"
    default_clabel = f'N PMT contrib. to peaks of pos(+)/neg(-) area / {sec_label}s'
    label = default_clabel if clabel is None else clabel
    cbar.set_label(label=label, fontsize=15)

    # pmt nr labels
    yticks = list(pmt_bin_e[1::3])
    ytick_l = [f"PMT {i}     " for i in range(n_pmts)]

    # neg/pos area labels
    yticks += [3*n for n in range(n_pmts)] + [3*n+1 for n in range(n_pmts)]
    ytick_l += ['-' for n in range(n_pmts)] + ['+' for n in range(n_pmts)]

    plt.yticks(yticks, ytick_l)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
