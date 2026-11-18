
prod = r'''
import sys, os, time
sys.path.insert(0, '/work')
import numpy as np
import wolff

L = 16; N2 = L*L
TEMPS = [round(1.5 + 0.1*k, 1) for k in range(14)]
NCHAIN = 4
NSTEPS = 500000
NBURN = 10000
OUT = '/work/data'
os.makedirs(OUT, exist_ok=True)

def seed_for(T, c):
    return 100000 + int(round(T*10))*100 + c

def fname(T, c):
    return f"{OUT}/M_T{T:.1f}_c{c}.npy"

def run_task(T, c):
    seed = seed_for(T, c)
    M, _ = wolff.wolff_run(L, T, NSTEPS, seed=seed, nburn=NBURN)
    np.save(fname(T, c), M.astype(np.int16))
    return M

t_start = time.time()
times = []
stop = False
for c in range(NCHAIN):
    for T in TEMPS:
        if os.path.exists(fname(T, c)):
            continue
        elapsed = time.time() - t_start
        est = max(times)*1.3 if times else 60.0
        if elapsed + est > BUDGET:
            stop = True
            break
        t0 = time.time()
        M = run_task(T, c)
        dt = time.time() - t0
        times.append(dt)
        m = M.astype(np.float64)/N2
        m2 = m*m
        U = 1 - (m2*m2).mean()/(3*m2.mean()**2)
        print(f"T={T} chain={c} seed={seed_for(T,c)} steps={NSTEPS} time={dt:.1f}s U_chain={U:.5f}", flush=True)
    if stop:
        break
print("total elapsed", round(time.time()-t_start,1), "s; tasks done:", sum(os.path.exists(fname(T,c)) for c in range(NCHAIN) for T in TEMPS), "of", NCHAIN*len(TEMPS))
'''
with open('/work/prod.py', 'w') as f:
    f.write(prod)
BUDGET = 400.0
exec(open('/work/prod.py').read())
