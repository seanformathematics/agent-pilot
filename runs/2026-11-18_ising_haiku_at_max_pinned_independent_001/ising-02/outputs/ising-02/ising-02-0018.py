
import numpy as np, json
TS = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]

def jk_from_sums(a2, a4, K):
    R = a2.size
    S2 = a2.sum(); S4 = a4.sum(); Nt = R*K
    U = 1 - (S4/Nt)/(3*(S2/Nt)**2)
    S2j = S2 - a2; S4j = S4 - a4; Nj = (R-1)*K
    Uj = 1 - (S4j/Nj)/(3*(S2j/Nj)**2)
    err = np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return float(U), float(err)

# extra T=2.5 runs with only U values (from logs, 5-decimal rounding)
extra25 = [0.38390, 0.38237, 0.38312, 0.38230, 0.38343, 0.38267, 0.38299, 0.38269, 0.38324, 0.38331]

final = {}
calib_num = 0.0; calib_den = 0.0
for T in TS:
    runs = []
    M1 = np.load(f"/work/prod_M_T{T:.2f}.npy").astype(np.float64)/256.0
    K1 = M1.shape[1]
    runs.append(dict(name='prod1', a2=(M1*M1).sum(1), a4=(M1**4).sum(1), K=K1))
    d = np.load(f"/work/prod2_sums_T{T:.2f}.npz"); runs.append(dict(name='prod2', a2=d['a2'], a4=d['a4'], K=int(d['K'])))
    if T >= 2.2:
        for nm in ['runA', 'runB']:
            dd = np.load(f"/work/{nm}_sums_T{T:.2f}.npz"); runs.append(dict(name=nm, a2=dd['a2'], a4=dd['a4'], K=int(dd['K'])))
    a2c = np.concatenate([r['a2'] for r in runs]); a4c = np.concatenate([r['a4'] for r in runs])
    Uc, ec = jk_from_sums(a2c, a4c, K1)
    Ui = []; ji = []
    for r in runs:
        u, e = jk_from_sums(r['a2'], r['a4'], r['K']); Ui.append(u); ji.append(e)
    Ui = np.array(Ui); ji = np.array(ji)
    n_runs = len(runs); n_samp = sum(r['a2'].size*r['K'] for r in runs)
    entry = dict(T=T, U_pooled=Uc, jk_pooled=ec, n_runs=n_runs, n_samples=int(n_samp), run_U=Ui.tolist(), run_jk=ji.tolist())
    if T == 2.5:
        allU = np.concatenate([Ui, np.array(extra25)])
        entry.update(n_runs=len(allU), n_samples=int(len(allU)*1000*K1), U_all14=float(allU.mean()),
                     sd_runs=float(allU.std(ddof=1)), se_emp=float(allU.std(ddof=1)/np.sqrt(len(allU))),
                     se_jk_equiv=float(ji.mean()/np.sqrt(len(allU))))
    if n_runs >= 2 and T >= 2.2 and T != 2.5:
        calib_num += np.sum((Ui-Ui.mean())**2); calib_den += (n_runs-1)*np.mean(ji)**2
    final[str(T)] = entry

# calibration over T=2.2..2.8 excluding 2.5 (4 runs each), plus 2.5 with its 14 runs using jk of its 4 runs
c2 = calib_num/calib_den
# add T=2.5 contribution
allU25 = np.array(final['2.5']['run_U'] + extra25)
calib_num += np.sum((allU25-allU25.mean())**2); calib_den += (len(allU25)-1)*np.mean(final['2.5']['run_jk'])**2
c2_all = calib_num/calib_den
print(f"calibration c^2 (T=2.2-2.8 excl 2.5): {c2:.3f}; including 2.5 with 14 runs: {c2_all:.3f} => c = {np.sqrt(c2_all):.3f}")
for T in TS:
    e = final[str(T)]
    line = f"T={T:.1f}: U_pooled={e['U_pooled']:.5f} jk_pooled={e['jk_pooled']:.6f} runs={e['n_runs']} n={e['n_samples']} run_U={np.round(e['run_U'],5)}"
    if T == 2.5:
        line += f" | 14-run mean={e['U_all14']:.5f} sd={e['sd_runs']:.5f} se_emp={e['se_emp']:.5f} se_jk_equiv={e['se_jk_equiv']:.5f}"
    print(line)
json.dump(final, open("/work/final_assembly.json", "w"), indent=1)
