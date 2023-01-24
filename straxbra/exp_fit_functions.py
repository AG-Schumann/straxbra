import numpy as np
from scipy.optimize import curve_fit

f_event_pars = ["t_\mathrm{{S11}}", "t_\mathrm{{decay}}", "t_\mathrm{{drift'}}", "\\tau", "a", "\\sigma", "A_\\mathrm{{S11}}", "A_\\mathrm{{S12}}", "A_\\mathrm{{S21}}", "A_\\mathrm{{S22}}", "dt_\\mathrm{{offset}}"]

f_event_exp_pars = ["t_0", "\\tau", "a", "A"]
f_event_gauss_pars = ["t_0", "\\sigma", "A"]

f_event_txt = ["t_S11", "t_decay", "t_drift", "tau", "a", "sigma", "A1", "A2", "A3", "A4", "dct_offset"]
f_de_txt = ["t_S11", "t_decay", "tau", "a", "A1", "A2"]
f_dg_txt = ["t_drift", "t_decay", "sigma", "A3", "A4", "dct_offset"]


def print_results(fit, fit_S1=False, fit_S2=False, p0 = False):
    if p0 is False:
        p0 = []
    if fit_S1 is False:
        fit_S1 = []
    if fit_S2 is False:
        fit_S2 = []
    print(f"{' '*12} {'p0':>8} {'fit':>8} {'fit_S1':>8} {'fit_S2':>8}")
    v_def = {l:" "*8 for l in f_event_txt}
    
    v0_d = {**v_def, **{l:f"{v:8.1f}" for l, v in zip(f_event_txt, p0)}}
    v1_d = {**v_def, **{l:f"{v:8.1f}" for l, v in zip(f_de_txt, fit_S1)}}
    v2_d = {**v_def, **{l:f"{v:8.1f}" for l, v in zip(f_dg_txt, fit_S2)}}
    
    for l, v in zip(f_event_txt, fit):
        print(f"{l:>12} {v0_d[l]} {v:8.1f} {v1_d[l]} {v2_d[l]}")




def print_p0_outa_bounds(p0, bounds, pars = f_event_txt):
    try:
        bl, bu = bounds

        for par, p, l, u in zip(pars, p0, bl, bu):
            if (p < l):
                print(f"\t\t{par} to low: {p:2g} < {l:2g}")
            if (p > u):
                print(f"\t\t{par} to high: {p:2g} > {u:2g}")
        return(None)
    except Exception:
        return(None)

def build_event_waveform(ps, baseline_spacing = 10):
    t0 = ps[0]["time"]

    ets = np.array([])
    ewf = np.array([])


    for p in ps:
        t_offs = p["time"]-t0

        t_end = max([0, *ets])
        dt = t_offs - t_end
        if (dt > 4 * baseline_spacing) and (baseline_spacing > 0):
            t_insert = np.arange(t_end+baseline_spacing, t_offs-baseline_spacing , baseline_spacing)
            dataz = np.zeros_like(t_insert)
            ets = np.append(ets, t_insert)
            ewf = np.append(ewf, dataz)

        ets = np.append(ets, t_offs+np.arange(0,p["length"])*p["dt"])
        # important norm peaks to PE/ns instead of PE/sample
        # some wide peaks have more than 10 ns sample width
        # this helps us with integration later
        ewf = np.append(ewf, p["data"][:p["length"]]/p["dt"])
    return(ets, ewf)

def f_event_exp(t, t_0, tau, a, A):
    return(A * 1/(1+np.exp((t_0-t)/a)) * np.exp((t_0-t)/tau))


def f_event_gauss(t, t_0, sigma, A):
    return(A * np.exp(-((t-t_0)**2/(2*sigma**2))))


def f_event_sum_exp(t, t_S11, t_decay, tau, a, A1, A2):
    t_S12 = t_S11 + t_decay
    return(
          f_event_exp(t, t_S11, tau, a, A1) 
        + f_event_exp(t, t_S12, tau, a, A2)
    )

def f_event_sum_gauss(t, t_S21, t_decay, sigma, A3, A4):
    t_S22 = t_S21 + t_decay
    return(
          f_event_gauss(t, t_S21, sigma, A3) 
        + f_event_gauss(t, t_S22, sigma, A3*A4)
    )


def f_event(t, t_S11, t_decay, t_drift, tau, a, sigma, A1, A2, A3, A4, dct_offset):
    
    t_S21 = t_S11 + t_drift
    
    return(
          f_event_sum_exp(t, t_S11, t_decay, tau, a, A1, A2)
        + f_event_sum_gauss(t, t_S21, t_decay+dct_offset, sigma, A3, A4)
    )


sep_props = {
    "event": (f_event, f_event_pars),
    "s1": (f_event_exp, f_event_exp_pars),
    "s2": (f_event_gauss, f_event_gauss_pars),
}


def f_event_p0(ets, ewf, ps, t_decay_default = 1500, **kwargs):
    t0 = ps["time"][0]
    ts = ps["time"] - t0

    try:
        ids_s1_pot = np.nonzero(ps["area"] < 500)[0]
        ps_S1s = ps[ids_s1_pot]
        peak_S11 = ps[ids_s1_pot[0]]
    except Exception:
        ps_S1s = ps[0]
        peak_S11 = ps_S1s[0]


    t_S11 = peak_S11["time"] - t0 + peak_S11["time_to_midpoint"]

    # find second S1 within 2500 ns
    ids_pot_s12 = [*np.nonzero((ps_S1s[1:]["time"] - peak_S11["time"]) < 2500)[0]+1, 0]
    
    
    id_pot_s12 = ids_s1_pot[ids_pot_s12[0]]

    peak_S12 = ps[id_pot_s12]
    if peak_S12["time"] == peak_S11["time"]:
        # first and second S1 peak is same peak
        try:
            # find the last 
            id_largest = np.argsort(peak_S11["data"])[::-1]
            id_diff = np.abs(np.diff(id_largest))
            id_first_offset = np.nonzero(id_diff > 10)[0][0]
            id_second_peak = id_largest[id_first_offset+1]
            t_decay = id_second_peak * peak_S11["dt"] - t_S11
        except:
            # fallback very high as low p0s tend to merge both S1 peaks during fit
            t_decay = t_decay_default
    else:
        # two distinct peaks found
        t_decay = peak_S12["time"] - peak_S11["time"]

    # S1s Done now S2(s)

    if len(ps) > 1:
        ps_ = ps[1:]
    else:
        ps_ = ps

    id_first_wide_peak = np.argmax(ps_["width"][:,5])
    widest_peak = ps_[id_first_wide_peak]

     
    
    t_drift = abs((widest_peak["time"]-t0)+widest_peak["time_to_midpoint"]-t_S11)

    id_widest_peak = np.argmax(ps_["width"][:,5])
    widest_peak = ps_[id_widest_peak]



    if t_S11 < 0:
        t_S11 = 0
    if t_decay < 0:
        t_decay = t_decay_default
    if t_drift < 0:
        t_drift = 100



    tau = 25
    a = 15
    sigma = widest_peak["width"][5]/2
    A1 = 5
    A2 = 2
    A3 = max(widest_peak["data"]/widest_peak["dt"])
    A4 = .25
    dct_offset = 0


    return(np.array([t_S11, t_decay, t_drift, tau, a, sigma, A1, A2, A3, A4, dct_offset]))



def f_event_bounds(ets, ewf, ps):
    inf = np.inf
    max_width = 10*np.max(ps["width"])
# t_S0, dt, tau, a, A1, A2
# 0, 1, 3, 4, 6, 7
#                 t_S11, t_decay, t_drift, tau,   a, sigma,   A1,  A2,  A3,  A4, t_offset
    l = np.array([    0,       0,       0,   0,   0,     0,    0,   0,   0,   0,    -2500])
    u = np.array([  inf,     inf,     inf, inf,  25,   inf,  inf, inf, inf,   1,     2500])
    return((l,u))

def extract_bounds(bounds, ids):
    l = np.array(bounds[0])[np.array(ids)]
    u = np.array(bounds[1])[np.array(ids)]
    return((l, u))
        



def fit_full_event(ets, ewf, ps, **kwargs):
    '''
    returns fit, sfit, p0, bounds
    '''
    fit = False
    sfit = False
    p0 = False
    bounds = False
    
    try:
        p0 = f_event_p0(ets, ewf, ps, **kwargs)
        bounds = f_event_bounds(ets, ewf, ps)
        
        ids_all = range(len(p0))

        t_S11, t_decay, t_drift, tau, a, sigma, A1, A2, A3, A4, dct_offset = p0
        f_fit = lambda t, t_S11, t_decay, t_drift, tau, a, sigma, A1, A2, A3, A4: f_event(t, t_S11, t_decay, t_drift, tau, a, sigma, A1, A2, A3, A4, dct_offset)

        pars_all = np.array(f_event.__code__.co_varnames[1:])
        pars_fit = np.array(f_fit.__code__.co_varnames[1:])
        id_to_use = [np.nonzero(pars_all == pf)[0][0]for pf in pars_fit]
        
        
        p0_fit = p0[id_to_use]
        bounds_fit = (
            bounds[0][id_to_use],
            bounds[1][id_to_use],
        )
        
        fit_, cov = curve_fit(
                f_fit,
                ets,
                ewf,
                p0 = p0_fit,
                absolute_sigma=True,
                bounds = bounds_fit,
                # full_output = True,

            )
        sfit_ = np.diag(cov)**.5
        
        fit = p0*1
        sfit = [-1]*len(p0)
        for i, v, sv in zip(id_to_use, fit_, sfit_):
            fit[i] = v
            sfit[i] = sv
    
    except Exception as e:
        print(f" Event fit failed: {e}")
        print_p0_outa_bounds(p0, bounds)
    return(fit, sfit, p0, bounds)


def fit_S1s(ets, ewf, fit, bounds):
    t_S11, t_decay, t_drift, tau, a, sigma, A1, A2, A3, A4, dct_offset = fit
    
    
    fit = False
    sfit = False
    
    try:
        p0 = (t_S11, t_decay, tau, a, A1, A2)
        bounds = extract_bounds(bounds, [0, 1, 3, 4, 6, 7])
        
        idx_S1 = np.nonzero((ets > t_S11 - 5 * tau) & (ets < t_S11 +t_decay+ 5*tau))[0]
        ets_S1 = ets[idx_S1]
        ewf_S1 = ewf[idx_S1]
    
    
        fit, cov = curve_fit(
            f_event_sum_exp,
            ets_S1,
            ewf_S1,
            p0 = p0,
            # bounds = bounds,
            absolute_sigma=True,
        )
        sfit = np.diag(cov)**.5
    
    except Exception as e:
        print(f" S1-fit failed: {e}")
        # print_p0_outa_bounds(p0, bounds, f_de_txt)
    return(fit, sfit)



def fit_S2s(ets, ewf, fit, fit_S1, bounds):
    t_S11, t_decay, t_drift, tau, a, sigma, A1, A2, A3, A4, dct_offset = fit
    t_S11, t_decay, tau, a, A1, A2 = fit_S1
    t_S21 = t_S11 + t_drift
    
    fit = False
    sfit = False
    
    try:
        p0 = (t_S21, t_decay, sigma, A3, A4)
        
        idx_S2 = np.nonzero((ets > t_S21 - 5 * sigma) & (ets < t_S21+t_decay+ 5*sigma))[0]
        ets_S2 = ets[idx_S2]
        ewf_S2 = ewf[idx_S2]
    
    
        fit, cov = curve_fit(
            f_event_sum_gauss,
            ets_S2,
            ewf_S2,
            p0 = p0,
            # bounds = bounds,
            absolute_sigma=True,
        )
        sfit = np.diag(cov)**.5
    
    except Exception as e:
        print(f" S2-fit failed: {e}")
        # print_p0_outa_bounds(p0, bounds, f_dg_txt)
    return(fit, sfit)

def get_aw(f, pars, dt = .5):
    '''
    return(area, width, t_mid, t_max)
    '''
    t0 = pars[0] - 5*pars[1]
    t1 = pars[0] + 5*pars[1]
    
    t = np.arange(t0, t1, dt)
    y = f(t, *pars)
    
    
    yf = f(t, *pars)
    
    area = np.sum(yf*dt)
    af = np.cumsum(yf*dt)/area

    t0, t_mid, t1 = np.interp([.25, .5, .75], af, t)
    width = t1-t0
    t_max = t[np.argmax(yf)]
    
    return(area, width, t_mid, t_max)

def extract_info(fit = False, fit_S1=False, fit_S2=False):
    if (fit is not False):
        t_S11, t_decay, t_drift, tau, a, sigma, A1, A2, A3, A4, dct_offset = fit
        suffix = False
        t_S21 = t_S11 + t_drift
    if (fit_S1 is not False) and (fit_S2 is not False):
        t_S11, t_decay, t_drift, tau, a, sigma, A1, A2, A3, A4, dct_offset = fit
        t_S21, t_decay, sigma, A3, A4 = fit_S2
        t_S11, t_decay, tau, a, A1, A2 = fit_S1
        suffix = "_2"
        t_drift = t_S21 - t_S11
    t_S12 = t_S11 + t_decay
    t_S22 = t_S21 + t_decay + dct_offset
    
    r = {
        f"decaytime": t_decay,
        "drifttime": t_drift/1000,
        "areas":[-1]*4,
        "widths":[-1]*4,
        "t_mid":[-1]*4,
        "t_max":[-1]*4,
        
    }
    for i, (f, pars) in enumerate([
        (f_event_exp, (t_S11, tau, a, A1)),
        (f_event_exp, (t_S12, tau, a, A2)),
        (f_event_gauss, (t_S21, sigma, A3)),
        (f_event_gauss, (t_S22, sigma, A3*A4)),
    ]):
        area, width, t_mid, t_max = get_aw(f, pars)
        r["areas"][i] = area
        r["widths"][i] = width
        r["t_mid"][i] = t_mid
        r["t_max"][i] = t_max
    
    if suffix is not False:
        r = {f"{n}{suffix}":v for n,v in r.items()}
    
    return(r)




### functions here are for second plugin (fits S2s only in sp_krypton)


def sum_gauss(t, t_S21, t_decay, sigma, A1, A2):
    return(
          f_event_gauss(t, t_S21,         sigma, A1)
        + f_event_gauss(t, t_S21+t_decay, sigma, A2)
    )

def fit_S2(ets, ewf, t_decay, S21_width):
    t_S21 = ets[np.argmax(ewf)]
    sigma = S21_width/2
    A1 = max(ewf)
    A2 = A1 * .15
    
    p0 = [t_S21, sigma, A1, A2]
    
    # use this fancy constric to fix t_decay to the result from the S1s
    f_fit = lambda t, t_S21, sigma, A1, A2: sum_gauss(t, t_S21, t_decay, sigma, A1, A2)
    
    
    fit, cov = curve_fit(
        f_fit,
        ets, ewf,
        p0 = p0,
        absolute_sigma=True,
        bounds = [
            (0, 0, 0, 0),
            (np.inf, np.inf, np.inf, np.inf)
        ]
    )
    sfit = np.diag(cov)**.5
    
    # this way we can throw fit into our original function
    fit = np.insert(fit, 1,t_decay)
    sfit = np.insert(sfit, 1,0)
    return(fit, sfit)

