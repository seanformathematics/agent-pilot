
import json
res = json.load(open('/work/final_results.json'))
for r in res:
    print(f"T={r['T']}: U={r['U']!r}, U_err={r['U_err']!r}, n={r['n_samples']}")
