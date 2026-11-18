
import numpy as np, json
fa = json.load(open("/work/final_assembly.json"))
c = 1.23
TS = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
points = []
for T in TS:
    e = fa[str(T)]
    if T == 2.5:
        U = e['U_all14']; Uerr = e['se_emp']; n = e['n_samples']
    else:
        U = e['U_pooled']; Uerr = c*e['jk_pooled']; n = e['n_samples']
    points.append(dict(T=T, U=float(U), U_err=float(Uerr), n_samples=int(n)))
    print(f"T={T}: U={U:.6f}  U_err={Uerr:.8f}  n={n}")
# seeds actually used in final estimates
seeds = []
seeds += [900000 + 7*i for i in range(14)]          # prod1 (all T)
seeds += [7000 + i for i in range(14)]              # prod2 (all T)
seeds += [8000 + 10*i + 70 - 70 + 0 for i in []]    # placeholder (no-op)
seeds += [8000 + 10*i for i in range(7, 14)]        # runA (T=2.2..2.8)
seeds += [8001 + 10*i for i in range(7, 14)]        # runB (T=2.2..2.8)
seeds += [3101, 3102, 3103, 3104, 5101, 5102, 5103, 5104, 5105, 5106]  # extra T=2.5 runs
print("n seeds", len(seeds), "unique", len(set(seeds)))
print(json.dumps(points))
print(json.dumps(seeds))
