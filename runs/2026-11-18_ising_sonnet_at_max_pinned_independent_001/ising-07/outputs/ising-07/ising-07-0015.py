
import sys, json
sys.path.insert(0,'/work')
import numpy as np
from prod import *

Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
final=[]; seeds=[]
print("T    U_impr       err_chain    err_block    err_used     U_direct     err_direct   n_samples  N_eff(approx)  tau_i")
for T in Ts:
    z=np.load("/work/prod_T%.1f.npz"%T)
    blocks=z['blocks']; K,nb,R,_=blocks.shape
    steps=int(z['steps']); neq=int(z['neq'])
    seeds += [int(s) for s in z['seeds']]
    pooled=blocks.transpose(1,0,2,3).reshape(nb,K*R,4)
    ai=analyze(pooled,'i'); ad=analyze(pooled,'d')
    err=max(ai['err_chain'],ai['err_block'])
    n=K*R*steps
    ti,_=tau_int(z['acf_i'].mean(axis=0))
    neff=n/(2*ti)
    final.append(dict(T=T,U=float(ai['U']),U_err=float(err),n_samples=int(n),U_direct=float(ad['U']),U_direct_err=float(ad['err_chain']),tau_int=float(ti)))
    print("%.1f  %.8f  %.8f  %.8f  %.8f  %.8f  %.8f  %d  %.0f  %.2f"%(T,ai['U'],ai['err_chain'],ai['err_block'],err,ad['U'],ad['err_chain'],n,neff,ti))
json.dump(dict(points=final,seeds=seeds,N=16,K=K,R=R,neq=neq,steps=steps),open('/work/final_results.json','w'),indent=1)
print("max U_err = %.6f (target 0.005)"%max(f['U_err'] for f in final))
print("min U = %.5f, max U = %.6f (must be <= 2/3 = 0.666667)"%(min(f['U'] for f in final),max(f['U'] for f in final)))
print("n seeds:",len(seeds), "first/last:",seeds[0],seeds[-1])
