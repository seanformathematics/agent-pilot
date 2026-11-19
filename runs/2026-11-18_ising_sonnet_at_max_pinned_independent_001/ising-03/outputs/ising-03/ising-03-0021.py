
import sys, pickle
sys.path.insert(0,'/work')
import numpy as np
import sw
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def ld(fn): return pickle.load(open(fn,'rb'))
def homog(b2):
    R,nb=b2.shape
    pr=b2.mean(axis=1)
    v_between=pr.var(ddof=1)                        # variance across replicas of the per-replica mean
    v_within=(b2.var(axis=1,ddof=1)/nb).mean()      # expected if blocks are independent
    return v_between/v_within
runs={}
for T in (2.5,2.8):
    k=Ts.index(T)
    runs[(T,'old Metro cold s33000+k')]=ld(f'/work/metro_k{k}.pkl')['blk']
    if T==2.5:
        runs[(T,'new Metro hot  s55002')]=ld(f'/work/metro2_k{k}.pkl')['blk']
        runs[(T,'new Metro cold s77001')]=ld(f'/work/metro3_cold_k{k}.pkl')['blk']
        runs[(T,'new Metro hot  s77002')]=ld(f'/work/metro3_hot_k{k}.pkl')['blk']
    else:
        runs[(T,'new Metro hot  s55001')]=ld(f'/work/metro2_k{k}.pkl')['blk']
    A=ld(f'/work/prodA_k{k}.pkl')['blk']; B=ld(f'/work/prodB_k{k}.pkl')['blk']
    runs[(T,'SW A')]=A[0:2]; runs[(T,'SW B')]=B[0:2]
for (T,name),blk in runs.items():
    b2=blk[0]; b4=blk[1]
    R,nb=b2.shape
    pr2=b2.mean(axis=1)
    # 10 groups of replicas
    gm=[pr2[i*(R//10):(i+1)*(R//10)].mean() for i in range(10)]
    U,e=sw.U_jack(pr2,b4.mean(axis=1))
    # skew/kurtosis of per-replica mean m2
    z=(pr2-pr2.mean())/pr2.std(); sk=(z**3).mean(); ku=(z**4).mean()
    # between/within variance ratio for m2 blocks and for m4 blocks
    print(f"T={T} {name:24s} R={R} nb={nb}: U={U:.5f}+-{e:.5f}  m2={pr2.mean():.5f}  var_between/var_within(m2)={homog(b2):.2f} (m4)={homog(b4):.2f}  skew={sk:+.2f} kurt={ku:.2f}  group means (x1e3)={np.round(np.array(gm)*1e3,1)}")
