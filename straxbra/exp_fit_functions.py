import numpy as np
from scipy.optimize import curve_fit

f_event_pars = ["t_\mathrm{{S11}}", "t_\mathrm{{decay}}", "t_\mathrm{{drift'}}", "\\tau", "a", "\\sigma", "A_\\mathrm{{S11}}", "A_\\mathrm{{S12}}", "A_\\mathrm{{S21}}", "A_\\mathrm{{S22}}", "dt_\\mathrm{{offset}}"]

f_event_exp_pars = ["t_0", "\\tau", "a", "A"]
f_event_gauss_pars = ["t_0", "\\sigma", "A"]

f_event_txt = ["t_S11", "t_decay", "t_drift", "tau", "a", "sigma", "A_S11", "A_S12", "A_S21", "A_S22", "dt_off"]
f_de_txt = ["t_S11", "t_decay", "tau", "a", "A1", "A2"]
f_dg_txt = ["t_S11", "t_decay", "t_drift", "sigma", "A3", "A4"]

def print_p0_outa_bounds(p0, bounds, pars = f_event_txt):
    try:
        bl, bu = bounds

        for par, p, l, u in zip(pars, p0, bl, bu):
            if (p < l):
                print(f"{par} to low: {p:2g} < {l:2g}")
            if (p > u):
                print(f"{par} to high: {p:2g} > {u:2g}")
        return(None)
    except Exception:
        return(None)

def build_event_waveform(ps):
    t0 = ps[0]["time"]

    ets = np.array([])
    ewf = np.array([])


    for p in ps:
        t_offs = p["time"]-t0

        t_end = max([0, *ets])
        dt = t_offs - t_end
        if dt > 25:
            t_insert = np.arange(t_end+10, t_offs-10 , 10)
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

def f_event_sum_gauss(t, t_S11, t_decay, t_drift, sigma, A3, A4):
    t_S21 = t_S11 + t_decay + t_drift
    t_S22 = t_S21 + t_decay
    return(
          f_event_gauss(t, t_S21, sigma, A3) 
        + f_event_gauss(t, t_S22, sigma, A3*A4)
    )


def f_event(t, t_S11, t_decay, t_drift, tau, a, sigma, A1, A2, A3, A4, dct_offset):
    return(
          f_event_sum_exp(t, t_S11, t_decay, tau, a, A1, A2)
        + f_event_sum_gauss(t, t_S11, t_decay+dct_offset, t_drift, sigma, A3, A4)
    )


sep_props = {
    "event": (f_event, f_event_pars),
    "s1": (f_event_exp, f_event_exp_pars),
    "s2": (f_event_gauss, f_event_gauss_pars),
}


def f_event_p0(ets, ewf, ps):

    t0 = ps["time"][0]
    ts = ps["time"] - t0
    
    id_s1_pot = np.nonzero(ps["area"] < 500)[0][0]
    t_S11 = ts[id_s1_pot]
    ps_ = ps[((ts-t_S11) >= 0) & ((ts - t_S11) < 50000) ]
    
    
    id_widest_peak = np.argmax(ps_["width"][:,5])
    widest_peak = ps_[id_widest_peak]
    
    t_decay = 150
    t_drift = abs((widest_peak["time"]-t0)-t_decay-t_S11)

    #     t_decay  = min(t_decay, 2500)
    #t_drift  = min(t_drift, 50000)
    
    tau = 25
    a = 10
    sigma = widest_peak["width"][5]/10
    A1 = 3
    A2 = 1
    A3 = max(ewf)
    A4 = .2
    dct_offset = 0
    return(t_S11, t_decay, t_drift, tau, a, sigma, A1, A2, A3, A4, dct_offset)


def f_event_bounds(ets, ewf, ps):
    max_width = 10*np.max(ps["width"])
# t_S0, dt, tau, a, A1, A2
# 0, 1, 3, 4, 6, 7
#     t_S11, t_decay, t_drift, tau,  a,     sigma,   A1,   A2,    A3,  A4
    l = (     0,      10,    100,   0,   0,        3,   .1,  .1,    .1,  .0, -10)
    u = (50_000,    1500, 50_000, 100,  20, max_width, 12.5,  10, 100,   .9, 10)
    return((l,u))
def extract_bounds(bounds, ids):
    l = np.array(bounds[0])[np.array(ids)]
    u = np.array(bounds[1])[np.array(ids)]
    return((l, u))
        



def fit_full_event(ets, ewf, ps):
    
    fit = False
    sfit = False
    p0 = False
    bounds = False
    
    try:
        p0 = f_event_p0(ets, ewf, ps)
        bounds = f_event_bounds(ets, ewf, ps)
        
        
        fit, cov = curve_fit(
            f_event,
            ets,
            ewf,
            p0 = p0,
            bounds = bounds,
            absolute_sigma=True,
        )
        sfit = np.diag(cov)**.5
    
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
            bounds = bounds,
            absolute_sigma=True,
        )
        sfit = np.diag(cov)**.5
    
    except Exception as e:
        print(f" S1-fit failed: {e}")
        print_p0_outa_bounds(p0, bounds, f_de_txt)
    return(fit, sfit)



def fit_S2s(ets, ewf, fit, fit_S1, bounds):
    t_S11, t_decay, t_drift, tau, a, sigma, A1, A2, A3, A4, dct_offset = fit
    t_S11, t_decay, tau, a, A1, A2 = fit_S1
    
    
    fit = False
    sfit = False
    
    try:
        p0 = (t_S11, t_decay, t_drift, sigma, A3, A4)
        bounds = extract_bounds(bounds, [0, 1, 2, 5, 8, 9])

        idx_S2 = np.nonzero((ets > t_S11 + t_drift+t_decay - 5 * sigma) & (ets < t_S11 +t_drift+2*t_decay+ 5*sigma))[0]
        ets_S2 = ets[idx_S2]
        ewf_S2 = ewf[idx_S2]
    
    
        fit, cov = curve_fit(
            f_event_sum_gauss,
            ets_S2,
            ewf_S2,
            p0 = p0,
            bounds = bounds,
            absolute_sigma=True,
        )
        sfit = np.diag(cov)**.5
    
    except Exception as e:
        print(f" S2-fit failed: {e}")
        print_p0_outa_bounds(p0, bounds, f_dg_txt)
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
    if (fit_S1 is not False) and (fit_S2 is not False):
        t_S11, t_decay, t_drift, sigma, A3, A4 = fit_S2
        t_S11, t_decay, tau, a, A1, A2 = fit_S1
        suffix = "_2"
    t_S12 = t_S11 + t_decay
    t_S21 = t_S12 + t_drift
    t_S22 = t_S21 + t_decay
    
    r = {
        f"decaytime": t_decay,
        "drifttime": (t_decay + t_drift)/1000,
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
        absolute_sigma=True
    )
    sfit = np.diag(cov)**.5
    
    # this way we can throw fit into our original function
    fit = np.insert(fit, 1,t_decay)
    sfit = np.insert(sfit, 1,0)
    return(fit, sfit)

