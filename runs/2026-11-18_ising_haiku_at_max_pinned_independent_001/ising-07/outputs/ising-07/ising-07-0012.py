import json, math
import numpy as np
d = json.load(open("/work/final_results.json"))
rows = d["final"]
c = 1.2
points = []
for r in rows:
    Uerr = c * r["U_err_rep_raw"]
    assert math.isfinite(r["U"]) and math.isfinite(Uerr)
    assert 0 < Uerr <= 0.005, (r["T"], Uerr)
    assert r["U"] <= 2/3 + 1e-12
    points.append(dict(T=float(r["T"]), status="ok", U=float(f"{r['U']:.8f}"),
                       U_err=float(f"{Uerr:.4e}"), n_samples=int(r["n_samples"])))
for p in points:
    print(p)
print("seeds A:", [r["seedA"] for r in rows])
print("seeds B:", [r["seedB"] for r in rows])
print("max U_err:", max(p["U_err"] for p in points))
json.dump(points, open("/work/submission_points.json", "w"), indent=1)
