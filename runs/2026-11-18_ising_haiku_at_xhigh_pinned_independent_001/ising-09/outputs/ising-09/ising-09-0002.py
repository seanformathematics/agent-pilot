
import os
os.makedirs('/work', exist_ok=True)
src = r'''
import numpy as np, json, os
N = 16
R = 1000
ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
EVEN = ((ii + jj) % 2 == 0)
ODD = ~EVEN

def _acc_table(T):
    tab = np.zeros(17)
    for d in range(-8, 9):
        tab[d + 8] = 1.0 if d <= 0 else np.exp(-d / T)
    return tab

def _sweep(s, rng, acc_tab):
    for mask in (EVEN, ODD):
        nb = (np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2))
        dE = 2 * s * nb
        p = acc_tab[dE + 8]
        u = rng.random(s.shape)
        flip = (u < p) & mask
        s[flip] *= -1
    return s

def fname(T):
    return '/work/state_T%.3f.npz' % T

def run(T, nburn, nmeas_new, seed):
    f = fname(T)
    acc = _acc_table(T)
    if os.path.exists(f):
        d = np.load(f)
        s = d['s'].astype(np.int64)
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(d['rng']))
        S2 = d['S2'].copy(); S4 = d['S4'].copy()
        nmeas = int(d['nmeas']); nsw = int(d['nsw'])
        series = d['series'].copy()
    else:
        rng = np.random.default_rng(seed)
        s = rng.choice(np.array([-1, 1], dtype=np.int64), size=(R, N, N))
        S2 = np.zeros(R); S4 = np.zeros(R)
        nmeas = 0; nsw = 0
        series = np.zeros((0, 2))
    if nsw == 0 and nburn > 0:
        for _ in range(nburn):
            s = _sweep(s, rng, acc)
        nsw = nburn
    new = np.zeros((nmeas_new, 2))
    for k in range(nmeas_new):
        s = _sweep(s, rng, acc)
        m = s.reshape(R, -1).mean(axis=1)
        m2 = m * m
        S2 += m2; S4 += m2 * m2
        new[k, 0] = m2.mean(); new[k, 1] = (m2 * m2).mean()
    nmeas += nmeas_new
    nsw += nmeas_new
    series = np.vstack([series, new])
    np.savez(f, s=s, rng=json.dumps(rng.bit_generator.state), S2=S2, S4=S4,
             nmeas=nmeas, nsw=nsw, series=series)
    return S2, S4, nmeas, series

def ubinder_stats(S2, S4, nmeas):
    Rr = len(S2)
    tot2 = S2.sum(); tot4 = S4.sum()
    U = 1 - (tot4 / (nmeas * Rr)) / (3 * (tot2 / (nmeas * Rr)) ** 2)
    t2 = tot2 - S2; t4 = tot4 - S4
    Uj = 1 - (t4 / (nmeas * (Rr - 1))) / (3 * (t2 / (nmeas * (Rr - 1))) ** 2)
    err = np.sqrt((Rr - 1) / Rr * np.sum((Uj - Uj.mean()) ** 2))
    return U, err
'''
with open('/work/ising.py', 'w') as f:
    f.write(src)

import sys, time
sys.path.insert(0, '/work')
import ising
import numpy as np
for T in [1.5, 2.269]:
    if os.path.exists(ising.fname(T)):
        os.remove(ising.fname(T))
    t0 = time.time()
    S2, S4, nm, series = ising.run(T, nburn=1000, nmeas_new=2000, seed=1000 + int(T * 1000))
    U, err = ising.ubinder_stats(S2, S4, nm)
    print("T=%g time=%.1fs U=%.5f err=%.5f nmeas=%d" % (T, time.time() - t0, U, err, nm))
    for a in range(0, len(series), 400):
        print("  window", a, "<m2>=%.4f <m4>=%.4f" % tuple(series[a:a + 400].mean(axis=0)))
