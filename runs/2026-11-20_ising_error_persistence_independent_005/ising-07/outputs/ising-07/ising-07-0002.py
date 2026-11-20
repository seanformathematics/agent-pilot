
code = r'''
import random, math, sys, time, json
import numpy as np

def run(T, seed, t_budget, N=16, nequil=20000, block=2000):
    rng = random.Random(seed)
    V = N*N
    # neighbour table
    nb = [[0]*4 for _ in range(V)]
    for x in range(N):
        for y in range(N):
            i = x*N+y
            nb[i][0] = ((x+1)%N)*N + y
            nb[i][1] = ((x-1)%N)*N + y
            nb[i][2] = x*N + (y+1)%N
            nb[i][3] = x*N + (y-1)%N
    s = [1 if rng.random()<0.5 else -1 for _ in range(V)]
    M = sum(s)
    padd = 1.0 - math.exp(-2.0/T)
    rnd = rng.random
    def cluster_step():
        nonlocal M
        i0 = int(rnd()*V)
        sgn = s[i0]
        s[i0] = -sgn
        stack = [i0]
        cnt = 1
        while stack:
            i = stack.pop()
            for j in nb[i]:
                if s[j] == sgn and rnd() < padd:
                    s[j] = -sgn
                    stack.append(j)
                    cnt += 1
        M -= 2*sgn*cnt
    t0 = time.time()
    for _ in range(nequil):
        cluster_step()
    # measure
    Ms = []
    nsteps = 0
    while True:
        for _ in range(block):
            cluster_step()
            Ms.append(M)
        nsteps += block
        if time.time()-t0 > t_budget:
            break
    arr = np.array(Ms, dtype=np.float64)/V
    return arr, time.time()-t0

if __name__ == "__main__":
    T = float(sys.argv[1]); seed=int(sys.argv[2]); tb=float(sys.argv[3])
    arr, el = run(T, seed, tb)
    np.save(f"/work/m_T{T}_s{seed}.npy", arr)
    m2=np.mean(arr**2); m4=np.mean(arr**4)
    print(T, seed, len(arr), el, 1-m4/(3*m2*m2))
'''
import os
os.makedirs('/work', exist_ok=True)
open('/work/wolff.py','w').write(code)
import subprocess
r = subprocess.run(['python3','/work/wolff.py','2.269','1','20'],capture_output=True,text=True,cwd='/work')
print(r.stdout, r.stderr[-2000:])
