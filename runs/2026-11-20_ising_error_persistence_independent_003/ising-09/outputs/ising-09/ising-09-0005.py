import numpy as np
Ts = [1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
def U_of(S2, S4):
    m2 = S2.mean(); m4 = S4.mean()
    return 1 - m4 / (3 * m2 * m2)
for T in Ts:
    d = np.load(f"/work/raw_T{T}.npz")
    S2, S4, s2, s4 = d["S2"], d["S4"], d["series2"], d["series4"]
    R = len(S2); MEAS = len(s2)
    U = U_of(S2, S4)
    # jackknife over replicas
    tot2, tot4 = S2.sum(), S4.sum()
    jk = np.array([1 - ((tot4 - S4[r]) / (R - 1)) / (3 * ((tot2 - S2[r]) / (R - 1)) ** 2) for r in range(R)])
    Ujk = R * U - (R - 1) * jk.mean()
    err = np.sqrt((R - 1) / R * np.sum((jk - jk.mean()) ** 2))
    # equilibration/stationarity check: halves
    h = MEAS // 2
    U1 = 1 - s4[:h].mean() / (3 * s2[:h].mean() ** 2)
    U2 = 1 - s4[h:].mean() / (3 * s2[h:].mean() ** 2)
    # integrated autocorrelation time of series2 (replica-averaged)
    x = s2 - s2.mean()
    ac = np.correlate(x, x, 'full')[len(x)-1:] / (x @ x)
    tau = 1 + 2 * np.sum(ac[1:200] * (np.abs(np.arange(1,200)) < 200))
    print(f"T={T} U={Ujk:.5f} err_jk={err:.5f} naive_err={np.std(U*0+jk)*np.sqrt(R-1):.5f} halves={U1:.4f},{U2:.4f} tau_int(m2)~{tau:.1f} <m2>={S2.mean()/MEAS:.4f}")
