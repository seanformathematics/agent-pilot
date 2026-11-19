
import json
final = json.load(open('/work/final_results.json'))
for r in final:
    print(f"T={r['T']!r:5}  U={r['U']:.8f}  U_err={r['U_err']:.4g}  n_samples={r['n_samples']}  tau_int(m2)~{r['tau_int_m2']:.2f}")
print("max U_err =", max(r['U_err'] for r in final))
