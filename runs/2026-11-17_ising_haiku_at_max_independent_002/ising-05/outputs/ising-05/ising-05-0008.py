
import os, sys, json
import numpy as np
os.chdir('/work'); sys.path.insert(0, '/work')
import wolff_py as W

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
NCH = 4; BL = 100; NBL = 5000
rows = []
for i, T in enumerate(Ts):
    d = np.load(f'/work/results/T{T:.2f}.npz')
    S2 = d['S2'].astype(np.int64); S4 = d['S4'].astype(np.int64)
    seeds = [int(x) for x in d['seeds']]
    n = float(len(S2)*BL)
    U_full = 1 - (S4.sum()/n)/(3*(S2.sum()/n)**2)
    U_jk_plain, err = W.jk_U(S2, S4, BL, J=100)
    # bias-corrected jackknife estimate
    J = 100; g = len(S2)//J
    G2 = S2[:g*J].reshape(J, g).sum(axis=1).astype(float)
    G4 = S4[:g*J].reshape(J, g).sum(axis=1).astype(float)
    nl = float((J-1)*g*BL); na = float(g*J*BL)
    Ui = 1 - ((G4.sum()-G4)/nl)/(3*((G2.sum()-G2)/nl)**2)
    U_bc = J*U_full - (J-1)*Ui.mean()
    Uc = []
    for c in range(NCH):
        a = S2[c*NBL:(c+1)*NBL]; b = S4[c*NBL:(c+1)*NBL]
        Uc.append(1 - (b.sum()/(NBL*BL))/(3*(a.sum()/(NBL*BL))**2))
    Uc = np.array(Uc)
    rows.append(dict(T=T, U=float(U_full), U_err=float(err), U_bias_corr=float(U_bc),
                     chain_U=[float(x) for x in Uc], chain_std_err=float(Uc.std(ddof=1)/np.sqrt(NCH)),
                     n_samples=int(n), seeds=seeds, sign_flips=int(d['nflip']),
                     U_err_J20=float(W.jk_U(S2, S4, BL, J=20)[1]), U_err_J200=float(W.jk_U(S2, S4, BL, J=200)[1])))
    r = rows[-1]
    print(f"T={T}: U={r['U']:.8f}  U_err={r['U_err']:.3e}  (J20 {r['U_err_J20']:.3e}, J200 {r['U_err_J200']:.3e}, chain {r['chain_std_err']:.3e})  "
          f"U_bc={r['U_bias_corr']:.8f}  n={r['n_samples']}  flips={r['sign_flips']}  chainU={[round(x,5) for x in Uc]}", flush=True)
json.dump(rows, open('/work/results/final_table.json', 'w'), indent=1)
print('seeds:', [r['seeds'] for r in rows][0], '...', [r['seeds'] for r in rows][-1])
print('max U_err', max(r['U_err'] for r in rows), 'min U', min(r['U'] for r in rows), 'max U', max(r['U'] for r in rows))
