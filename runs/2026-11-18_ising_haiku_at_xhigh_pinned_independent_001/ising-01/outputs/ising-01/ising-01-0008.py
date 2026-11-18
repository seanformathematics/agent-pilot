
import json, math
d=json.load(open('/work/prod_results.json'))
for T in sorted(d, key=float):
    r=d[T]
    print(T, r['seed'], repr(r['U']), repr(r['err']), repr(r['U_half1']), repr(r['U_half2']), all(math.isfinite(r[k]) for k in ['U','err']))
print("max err", max(v['err'] for v in d.values()))
