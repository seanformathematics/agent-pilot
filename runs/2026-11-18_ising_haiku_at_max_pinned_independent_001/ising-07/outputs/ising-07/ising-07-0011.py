import json, sys
import numpy as np
sys.path.insert(0, "/work")
rows = json.load(open("/work/final_table.json"))
rep = json.load(open("/work/repeat_check.json"))
ext = json.load(open("/work/extra_runs.json"))

# Empirical single-run scatter vs mean single-run jackknife error
calib = {}
for T in [2.3, 2.5, 2.8]:
    row = [r for r in rows if abs(r["T"] - T) < 1e-9][0]
    Us = [row["U_A"], row["U_B"]]
    js = [row["sA"], row["sB"]]
    key = f"{T:.1f}"
    if key in rep:
        for run in rep[key]["runs"][1:]:  # skip the production run (already counted as U_A)
            Us.append(run[2]); js.append(run[3])
    if key in ext:
        for run in ext[key]:
            Us.append(run["U"]); js.append(run["Uerr_jack"])
    Us = np.array(Us); js = np.array(js)
    s_emp = Us.std(ddof=1)
    ratio = s_emp / js.mean()
    calib[key] = dict(n_runs=len(Us), s_emp=float(s_emp), mean_jack=float(js.mean()), ratio=float(ratio))
    print(f"T={T}: n_runs={len(Us)} s_emp={s_emp:.6f} mean_single_jack={js.mean():.6f} ratio={ratio:.3f}")
ratios = np.array([v["ratio"] for v in calib.values()])
rms_ratio = np.sqrt(np.mean(ratios**2))
z_all = []
for r in rows:
    if r["sA"] > 0 or r["sB"] > 0:
        z_all.append(r["z"])
z_all = np.array(z_all)
rms_z = np.sqrt(np.mean(z_all**2))
print("RMS of empirical/jackknife ratios:", round(rms_ratio, 3), " RMS z over all T:", round(rms_z, 3))
c = 1.10   # calibration factor: between RMS z (1.10) and RMS ratio (~1.0-1.1); conservative choice
final = []
print(f"{'T':>5} {'U':>12} {'U_err':>12} {'U_err_raw(rep)':>15} {'U_err(B20)':>12} {'n_samples':>10}")
for r in rows:
    Ufin = r["U_pool"]
    s_rep = r["s_pool_rep"]
    Uerr = c*s_rep
    assert np.isfinite(Ufin) and Uerr > 0 and Uerr <= 0.005 and Ufin <= 2/3 + 1e-12
    final.append(dict(T=r["T"], U=float(Ufin), U_err=float(Uerr), U_err_rep_raw=float(s_rep),
                      U_err_B20=float(r["s_pool_B20"]), n_samples=int(r["n_chains"]*r["n_sweeps_kept"]),
                      seedA=r["seedA"], seedB=r["seedB"], U_A=r["U_A"], U_B=r["U_B"]))
    print(f"{r['T']:5.2f} {Ufin:12.8f} {Uerr:12.4e} {s_rep:15.4e} {r['s_pool_B20']:12.4e} {int(r['n_chains']*r['n_sweeps_kept']):10d}")
json.dump(dict(final=final, calib=calib, rms_ratio=float(rms_ratio), rms_z=float(rms_z), c=c),
          open("/work/final_results.json", "w"), indent=1)
print("saved final_results.json")
