import sys, time
sys.path.insert(0, "/work")
import numpy as np
import sw as swmod

# Re-implement a series-recording variant (same algorithm) to inspect equilibration and autocorrelation
src = open("/work/sw.py").read()
src2 = src.replace("def run_sw(L, T, R, n_therm, n_meas, seed, start=\"random\", max_iter_report=False):",
                   "def run_sw_series(L, T, R, n_therm, n_meas, seed, start=\"random\"):")
src2 = src2.replace("    M2s = np.zeros(R, dtype=np.int64)\n    M4s = np.zeros(R, dtype=np.int64)\n    for _ in range(n_meas):\n        S = step(S)\n        M = S.sum(axis=(1, 2), dtype=np.int64)\n        M2 = M * M\n        M2s += M2\n        M4s += M2 * M2\n    return M2s, M4s, n_iter_total",
                    "    series = np.zeros((n_meas, R), dtype=np.int16)\n    for k in range(n_meas):\n        S = step(S)\n        series[k] = S.sum(axis=(1, 2), dtype=np.int64)\n    return series, n_iter_total")
ns = {}
exec(compile(src2, "sw_series", "exec"), ns)
run_sw_series = ns["run_sw_series"]

def tau_int_from_series(series):
    # series: (n, R) of M values; use m^2 = M^2 normalized
    x = series.astype(float)**2
    x = x - x.mean(axis=0)
    n, R = x.shape
    maxlag = 60
    c0 = (x*x).mean(axis=0)
    rho = np.zeros(maxlag)
    for t in range(maxlag):
        rho[t] = ((x[:n-t]*x[t:]).mean(axis=0) / c0).mean()
    # integrated tau with simple window
    tau = 0.5 + np.sum(rho[1:])
    return tau, rho[:8]

for T in [1.5, 2.269, 2.8]:
    for start in ["random", "ordered"]:
        t0 = time.time()
        ser, nit = run_sw_series(16, T, R=256, n_therm=0, n_meas=400, seed=77, start=start)
        dt = time.time()-t0
        # U vs time: average over replicas in windows
        M2 = ser.astype(float)**2
        M4 = M2**2
        early = 1 - M4[:10].mean()/(3*M2[:10].mean()**2)
        late = 1 - M4[200:].mean()/(3*M2[200:].mean()**2)
        tau, rho = tau_int_from_series(ser)
        print(f"T={T} start={start}: U(first10 steps)={early:.4f}  U(steps200-400)={late:.4f}  tau_int(m^2)~{tau:.2f} rho(1..4)={np.round(rho[1:5],3)}  time={dt:.1f}s iters/step={nit/400:.1f}")
