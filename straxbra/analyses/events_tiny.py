import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from . import analysis_tools as at
import straxbra

@straxbra.tiny_analysis(requires=('events', 'event_basics',
                         'peaks'))
def event_waveform(events,
                   peaks,
                   event_nr=0,
                   xlabel='t since first peak [us]',
                   ylabel='area / ns [PE]',
                   title='',
                   peak_color='gray',
                   s1_color='b',
                   s2_color='g',
                   max_time=None,
                   is_log=False,
                   figsize=(10, 4),
                   **kwargs):
    """Plots peak waveforms of given event.

    Arguments for decorator:
    context -- straxbra context
    run_id  -- 5-digit zero padded run-id (str)


    Arguments:
    events  -- Provided by decorator (DataFrame)
    peaks   -- Provided by decorator (ndarray)

    Keyword arguments:
    event_nr   -- Index of event to plot (default 0, int)
    xlabel     -- Label of x-axis (str)
    ylabel     -- Label of y-axis (str)
    title      -- Title of Plot   (str)
    peak_color -- Plot color of peaks (default 'gray')
    s1_color   -- Plot color of main S1 (default 'b')
    s2_color   -- Plot color of main S2 (default 'g')
    max_time   -- Time in ns relative to first peaks start time
                  after which peaks are not plotted (int)
    is_log     -- Whether or not to plot y-axis logarithmicly (default False)
    figsize    -- mpl figsize (tuple)
    kwargs     -- Any kwargs plt.plot accepts, except color.
    """

    if 'color' in kwargs or 'c' in kwargs:
        raise ValueError('Give plot color via peak_color, s1_color and s2_color not color or c.')

    event = events.iloc[event_nr]
    peaks = peaks[(peaks['time'] >= event['time']) & (peaks['time'] < event['endtime'])]

    if max_time is not None:
        peaks = peaks[peaks['time'] <= peaks[0]['time'] + max_time]

    colors = {event['s1_index']: s1_color,
              event['s2_index']: s2_color}

    plt.figure(figsize=figsize)
    for idx, peak in enumerate(peaks):
        color = peak_color if idx not in colors else colors[idx]
        at.plot_peak(peak, t0=peaks[0]['time'], color=color, **kwargs)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if is_log:
        plt.ylim(0.1, None)
        plt.yscale('log')
    else:
        plt.axhline(0, c='k', alpha=0.2)



@straxbra.tiny_analysis(requires=('event_basics', 'event_krypton_basics'))
def area_width(
            events,
            x='s2_area',
            y='s2_range_50p_area',
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

    """Show 2D histogram of events versus s2_area(x) and s2_range_50p_area(y) (default).

    Arguments for decorator:
    context -- straxbra context
    run_id  -- 5-digit zero padded run-id (str)

    Arguments:
    events     -- Provided by decorator (ndarray)

    Keyword arguments:
    x          -- Field name in events that is plotted on x-axis (default 's2_area', str)
    y          -- Field name in events that is plotted on y-axis (default 's2_range_50p_area', str)
    cuts       -- Boolean array or list of indices to apply cuts to events
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

    if cuts is not None:
        if isinstance(cuts, int) or isinstance(cuts[0], int):
            events = events.iloc[cuts]
        else:
            events = events.loc[cuts]

    xbins = np.logspace(0, 6, 100) if xbins is None else xbins
    ybins = np.logspace(0, 5, 100) if ybins is None else ybins

    counts, _, _ = np.histogram2d(events[x], events[y], bins=[xbins, ybins])

    plt.figure(figsize=figsize)
    im = plt.pcolormesh(xbins, ybins, counts.T, cmap=cmap, norm=colors.LogNorm(), **kwargs)

    plt.title(title)
    plt.xlabel(xlabel if xlabel else x)
    plt.ylabel(ylabel if ylabel else y)

    if xlog: plt.xscale('log')
    if ylog: plt.yscale('log')
    plt.colorbar()
    plt.tight_layout()


