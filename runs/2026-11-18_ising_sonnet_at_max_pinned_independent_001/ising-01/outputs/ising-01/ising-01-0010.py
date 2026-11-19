
import json, math
res = json.load(open("/work/final_results.json"))
for r in res:
    assert all(math.isfinite(r[k]) for k in ["U", "U_err"]) and r["U"] <= 2/3 and r["U_err"] > 0
    print(f'{{"T": {r["T"]}, "status": "ok", "U": {r["U"]:.8f}, "U_err": {r["U_err"]:.4g}, "n_samples": {r["n_samples"]}}}')
print("max U_err:", max(r["U_err"] for r in res), " all <= 0.005:", all(r["U_err"] <= 0.005 for r in res))
print("autocorr-run seeds:", {T: 777 + int(T * 10) for T in [1.5, 2.0, 2.3, 2.6, 2.8]})
print("run1 seeds:", [1000 + int(round(T * 10)) for T in [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]])
print("run2 seeds:", [2000 + int(round(T * 10)) for T in [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]])
print("metropolis seeds:", [3000 + int(round(T * 10)) for T in [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]])
