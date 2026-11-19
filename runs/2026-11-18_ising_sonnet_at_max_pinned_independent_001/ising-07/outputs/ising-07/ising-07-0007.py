
import sys, time, os
sys.path.insert(0,'/work')
import numpy as np
from prod import *

Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
N=16; K=10; R=100; NEQ=300; STEPS=2000

def produce(idx_list):
    t0=time.time()
    for i in idx_list:
        T=Ts[i]
        blocks=[]; accd=[]; acci=[]; burns=[]; seeds=[]
        for k in range(K):
            seed = 100000 + 1000*i + k
            out = run_batch(N,T,R=R,neq=NEQ,steps=STEPS,seed=seed,record_burn=100)
            blocks.append(out['blocks']); accd.append(out['acf_d']); acci.append(out['acf_i'])
            b = out['burn']   # (100,R,2) hot chains even idx, cold chains odd idx
            burns.append(np.stack([b[:,0::2,:].mean(axis=1), b[:,1::2,:].mean(axis=1)],axis=1))  # (100,2[hot,cold],2[d,i])
            seeds.append(seed)
        np.savez("/work/prod_T%.1f.npz"%T, blocks=np.array(blocks), acf_d=np.array(accd), acf_i=np.array(acci),
                 burn=np.array(burns), seeds=np.array(seeds), T=T, N=N, R=R, neq=NEQ, steps=STEPS)
        print("T=%.1f done (%.0fs)"%(T,time.time()-t0), flush=True)

produce(range(0,7))
