import sys, os, time, json
sys.path.insert(0, '/work')
import numpy as np
from wolff_lib import wolff_series

t_start = time.time()
Ts = [round(1.5 + 0.1*i, 1) for i in range(14)]
K = 8; NUPD = 125000; NEQ = 5000; BUDGET = 430.0
seedlog = {}
done_now = 0
for iT, T in enumerate(Ts):
    for k in range(K):
        fname = f'/work/data/B4_T{T:.1f}_c{k}.npy'
        if os.path.exists(fname):
            continue
        if time.time() - t_start > BUDGET:
            break
        seed = 300000 + 1000*(iT+1) + k + 17
        M = wolff_series(16, T, seed, NUPD, NEQ)
        np.save(fname, M)
        seedlog[f'B4_T{T:.1f}_c{k}'] = seed
        done_now += 1
json.dump(seedlog, open(f'/work/data/seeds_B4_{int(time.time())}.json', 'w'), indent=1)
nfiles = len([f for f in os.listdir('/work/data') if f.startswith('B4_')])
print("batch4 chains done this call:", done_now, " total batch4 files:", nfiles, "of", len(Ts)*K)
print("elapsed", round(time.time()-t_start,1))
