
import glob, json, numpy as np
L=16; N2=L*L
TEMPS=[round(1.5+0.1*k,1) for k in range(14)]
res=json.load(open('/work/results.json'))
rng=np.random.default_rng(20240607)
out=[]
for T in TEMPS:
    files=sorted(glob.glob(f"/work/data/M_T{T:.1f}_c*.npy"))
    chains=[np.load(f).astype(np.int64) for f in files]
    B=50
    s2=[];s4=[];bl=[]
    for M in chains:
        m=M/N2; m2=m*m; m4=m2*m2
        bs=len(m)//B
        s2.append(m2[:bs*B].reshape(B,bs).sum(1)); s4.append(m4[:bs*B].reshape(B,bs).sum(1)); bl.append(np.full(B,bs,float))
    s2=np.concatenate(s2); s4=np.concatenate(s4); bl=np.concatenate(bl)
    nbk=len(s2)
    S2=s2.sum(); S4=s4.sum(); NN=bl.sum()
    U=1-(S4/NN)/(3*(S2/NN)**2)
    # bootstrap over blocks
    idx=rng.integers(0,nbk,size=(2000,nbk))
    b2=s2[idx].sum(1); b4=s4[idx].sum(1); bn=bl[idx].sum(1)
    Ub=1-(b4/bn)/(3*(b2/bn)**2)
    err_boot=Ub.std(ddof=1)
    e_jk=res[str(T)]['err'] if str(T) in res else res[repr(T)]['err']
    out.append((T,U,e_jk,err_boot,int(NN),res[str(T)]['K'] if str(T) in res else res[repr(T)]['K']))
    print(f"T={T:.1f}  U={U:.8f}  jk_err={e_jk:.3e}  boot_err={err_boot:.3e}  n={int(NN)}  K={res[str(T)]['K']}")
print("max jk err", max(o[2] for o in out), " max boot err", max(o[3] for o in out))
