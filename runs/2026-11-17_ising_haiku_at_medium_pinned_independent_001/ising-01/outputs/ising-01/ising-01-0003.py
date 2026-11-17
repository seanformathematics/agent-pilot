
import sys, time
sys.path.insert(0, '/work')
import numpy as np
from scipy.special import ellipk
from sw import *

def onsager_e(T):
    K = 1.0 / T
    k = 2*np.sinh(2*K)/np.cosh(2*K)**2
    K1 = ellipk(k*k)
    return -(1/np.tanh(2*K))*(1 + (2/np.pi)*(2*np.tanh(2*K)**2 - 1)*K1)

def analyze(d):
    m2 = d['m2']; m4 = d['m4']; cnt = d['cnt'].astype(float)
    R = len(m2)
    # per-replica means
    M2 = m2 / cnt; M4 = m4 / cnt
    U = 1 - M4.mean() / (3 * M2.mean()**2)
    # jackknife over replicas
    S2 = m2.sum(); S4 = m4.sum(); C = cnt.sum()
    Ul = 1 - ((S4 - m4) / (C - cnt)) / (3 * ((S2 - m2) / (C - cnt))**2)
    Uj = Ul.mean()
    err = np.sqrt((R - 1) / R * np.sum((Ul - Uj)**2))
    return U, err

t0 = time.time()
for T in [1.5, 2.0, 2.269, 2.8]:
    d = run(T, 16, 512, 100, 300, seed=12345)
    U, err = analyze(d)
    e = (d['e'] / d['cnt']).mean()
    print(f"T={T} U={U:.4f} err={err:.4f}  E/N^2 sim={e:.5f} onsager={onsager_e(T):.5f}  <|m|>={(d['am']/d['cnt']).mean():.4f}  time={time.time()-t0:.1f}")
