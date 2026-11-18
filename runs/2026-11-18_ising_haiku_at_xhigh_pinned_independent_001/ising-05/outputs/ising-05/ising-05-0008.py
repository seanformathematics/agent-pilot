
import json, math
d = json.load(open('/work/prod1.json')); d.update(json.load(open('/work/prod2.json')))
for T in sorted(d, key=float):
    v = d[T]
    print(T, repr(v['U']), repr(v['err']), repr(v['U_seed1']), repr(v['U_seed2']), math.isfinite(v['U']), v['err']>0)
