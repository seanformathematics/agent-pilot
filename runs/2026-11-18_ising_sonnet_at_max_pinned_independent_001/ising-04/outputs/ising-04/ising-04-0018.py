
import json
import numpy as np

cand = json.load(open("/work/final_candidates.json"))
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
SAFETY = 1.2
points = []
for T in Ts:
    c = cand[str(T)]
    base = max(c["jk200"], c["delta"], c["block"])
    points.append(dict(T=T, status="ok", U=float(c["U"]), U_err=float(SAFETY * base), n_samples=int(c["n"])))
    print(f"T={T}: U={c['U']:.6f}  U_err={SAFETY*base:.2e}  (jk200 {c['jk200']:.2e}, delta {c['delta']:.2e}, block {c['block']:.2e})  n={c['n']}")
print("max U_err =", max(p["U_err"] for p in points), " (target 0.005)")

seeds = []
seeds += [100000 + int(round(T * 10)) for T in Ts]                    # production run 1 (hot start)
seeds += [300000 + int(round(T * 10)) for T in Ts]                    # production run 2 (cold start)
seeds += [123]                                                        # timing test
seeds += [1000 + i for i in range(14)] + [5000 + i for i in range(14)] + [7000 + i for i in range(14)]   # N=4 exact checks, N=16 energy check
seeds += [20000 + i for i in range(14)]                               # pilot
seeds += [200000 + 1000 * int(round(T * 10)) + j for T in (1.6, 2.0, 2.3, 2.5, 2.8) for j in range(20)]  # calibration repeats
seeds += [400000 + int(round(T * 10)) for T in (2.0, 2.3, 2.7)]
seeds += [410000 + int(round(T * 10)) for T in (2.3, 2.4, 2.7, 2.8)]
seeds += [420000 + int(round(T * 10)) for T in (2.8, 2.7, 2.6, 2.5)]  # Metropolis cross-checks
print(len(seeds), "seeds; unique:", len(set(seeds)))
json.dump(dict(points=points, seeds=seeds), open("/work/submission.json", "w"))
print(json.dumps(points))
print(json.dumps(seeds))
