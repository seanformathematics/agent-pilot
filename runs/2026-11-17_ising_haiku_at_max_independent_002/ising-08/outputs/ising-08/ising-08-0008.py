import json, math
with open('/work/final_results.json') as fh:
    final = json.load(fh)
for r in final:
    print(f"T={r['T']:.2f}  U={r['U']!r}  U_err={r['U_err']!r}  n={r['n_samples']}  seeds={r['seeds']}")
print('all U_err > 0 and finite:', all(r['U_err'] > 0 and math.isfinite(r['U_err']) for r in final))
print('all U finite and < 2/3:', all(math.isfinite(r['U']) and r['U'] < 2/3 for r in final))
print('max U_err:', max(r['U_err'] for r in final))
