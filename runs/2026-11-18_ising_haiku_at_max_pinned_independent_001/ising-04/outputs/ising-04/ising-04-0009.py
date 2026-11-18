
import json
Ts_all = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
R = 4
seeds_all = []
per_T = {}
for T in Ts_all:
    idx = Ts_all.index(T)
    s = [100003*(idx+1) + 7*r + 1 for r in range(R)]
    per_T[T] = s
    seeds_all += s
print("production seeds:", seeds_all)
print("count:", len(seeds_all), "distinct:", len(set(seeds_all)))
tab = json.load(open("/work/final_table.json"))
pts = []
for row in tab:
    pts.append(dict(T=row["T"], status="ok", U=round(row["U"], 6), U_err=float("%.3e" % row["U_err"]), n_samples=row["n"]))
print(json.dumps(pts, indent=None))
