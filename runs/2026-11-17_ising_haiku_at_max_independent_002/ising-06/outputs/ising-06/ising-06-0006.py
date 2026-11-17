import sys, time, math, random
sys.path.insert(0, '/work')
import numpy as np
import ising

def wolff_traj(N, T, nsteps, seed, init_up=True):
    NN = N*N
    nbr = ising.build_nbr(N)
    rnd = random.Random(seed).random
    p = 1.0 - math.exp(-2.0/T)
    spins = [1]*NN if init_up else [1 if rnd() < 0.5 else -1 for _ in range(NN)]
    M = sum(spins)
    traj = np.empty(nsteps, dtype=np.int64)
    for k in range(nsteps):
        x0 = int(rnd()*NN)
        s0 = spins[x0]
        spins[x0] = -s0
        stack = [x0]; pop = stack.pop; push = stack.append
        c = 1
        while stack:
            x = pop()
            for y in nbr[x]:
                if spins[y] == s0 and rnd() < p:
                    spins[y] = -s0; push(y); c += 1
        M -= 2*s0*c
        traj[k] = M
    return traj

checkpoints = [0,1,2,5,10,20,50,100,200,500,999]
for T in [1.5, 2.27, 2.8]:
    t0 = time.time()
    nch = 400
    trajs = np.array([wolff_traj(16, T, 1000, seed=900000+k, init_up=True) for k in range(nch)])
    # long-run equilibrium reference from a random-start chain
    Mref, _ = ising.wolff_chain(16, T, 200000, 5000, seed=777)
    m2ref = np.mean(Mref.astype(float)**2); m4ref = np.mean(Mref.astype(float)**4)
    print(f"T={T}: equilibrium ref <M^2>={m2ref:.1f}, U={1-m4ref/(3*m2ref**2):.5f}, <M>={Mref.mean():.2f}")
    for c in checkpoints:
        col = trajs[:, c].astype(float)
        m2 = np.mean(col**2)
        m4 = np.mean(col**4)
        print(f"   t={c+1:4d} updates: <M^2>_ens={m2:9.1f} (rel dev {m2/m2ref-1:+.4f}), <M>_ens={col.mean():+8.2f}, U_ens={1-m4/(3*m2**2):.4f}")
    print(f"   time {time.time()-t0:.1f}s", flush=True)
