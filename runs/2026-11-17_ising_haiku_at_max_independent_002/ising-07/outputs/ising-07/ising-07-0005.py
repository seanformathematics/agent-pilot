import sys
sys.path.insert(0, '/work')
import wolff_prod as wp
info = wp.run_call(call_idx=2, seed=2002, n_burn=1000, n_rec_target=8000, time_budget=500.0)
print(info)
rows, ntot, nfiles = wp.analyze()
print("recorded steps per chain:", ntot, "block files:", nfiles)
for r in rows:
    print(f"T={r['T']:.2f} U={r['U']:.5f} Uerr={r['U_err']:.5f} n={r['n_samples']} U_h1={r['U_half1']:.5f} U_h2={r['U_half2']:.5f} <m2>={r['m2']:.5f} E={r['E']:.5f}+-{r['E_err']:.5f}")
