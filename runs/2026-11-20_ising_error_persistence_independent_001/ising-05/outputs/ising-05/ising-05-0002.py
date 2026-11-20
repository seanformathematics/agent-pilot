import os, time, math, random
os.makedirs('/work', exist_ok=True)
code = r'''
import math, random, sys, time
import numpy as np

def run(N, T, nsweeps_clusters, seed, nburn):
    rng = random.Random(seed)
    L = N*N
    # neighbors
    nb = []
    for i in range(N):
        for j in range(N):
            nb.append([((i+1)%N)*N+j, ((i-1)%N)*N+j, i*N+(j+1)%N, i*N+(j-1)%N])
    p = 1.0 - math.exp(-2.0/T)
    s = [1 if rng.random()<0.5 else -1 for _ in range(L)]
    rand = rng.random
    def cluster_flip():
        k = int(rand()*L)
        sk = s[k]
        s[k] = -sk
        stack = [k]
        size = 1
        while stack:
            x = stack.pop()
            for y in nb[x]:
                if s[y] == sk and rand() < p:
                    s[y] = -sk
                    stack.append(y)
                    size += 1
        return size
    for _ in range(nburn):
        cluster_flip()
    m2 = np.empty(nsweeps_clusters); m4 = np.empty(nsweeps_clusters); am=np.empty(nsweeps_clusters)
    for t in range(nsweeps_clusters):
        cluster_flip()
        M = sum(s)/L
        M2 = M*M
        m2[t] = M2; m4[t] = M2*M2
    return m2, m4

if __name__ == "__main__":
    N=int(sys.argv[1]); T=float(sys.argv[2]); n=int(sys.argv[3]); seed=int(sys.argv[4]); nb_=int(sys.argv[5])
    t0=time.time()
    m2,m4 = run(N,T,n,seed,nb_)
    print("time",time.time()-t0, "U", 1-m4.mean()/(3*m2.mean()**2))
'''
open('/work/wolff.py','w').write(code)
import subprocess
print(subprocess.run(['python3','/work/wolff.py','16','2.269','20000','1','2000'],capture_output=True,text=True,timeout=200).stdout)
