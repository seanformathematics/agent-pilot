
import json, numpy as np
d = json.load(open("/work/final_summary.json"))
seeds = []
for k, v in d.items():
    z = np.load(f"/work/prod_T{float(k):.1f}.npz")
    seeds += [int(z["seed_h"]), int(z["seed_c"])]
    print(f"{{\"T\": {float(k)}, \"status\": \"ok\", \"U\": {v['U']:.9f}, \"U_err\": {max(v['err'], 0):.9f}, \"n_samples\": {v['n']}}},   # blk err {v['err_blk']:.9f}, boot {v['err_boot']:.9f}")
print("seeds:", sorted(seeds))
print("max U_err:", max(v['err'] for v in d.values()))
print("all U in (0, 2/3):", all(0 < v['U'] < 2/3 for v in d.values()))
