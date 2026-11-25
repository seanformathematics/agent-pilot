import numpy as np, glob
def ujack(m2,m4):
    a2=m2.mean(0); a4=m4.mean(0); R=len(a2)
    U=1-a4.mean()/(3*a2.mean()**2)
    S2=a2.sum(); S4=a4.sum()
    x2=(S2-a2)/(R-1); x4=(S4-a4)/(R-1)
    loo=1-x4/(3*x2*x2)
    err=np.sqrt((R-1)/R*np.sum((loo-loo.mean())**2))
    return U,err
files={1.9:"run_T1.9_s200.npz",2.0:"run_T2.0_s201.npz",2.1:"run_T2.1_s202.npz",2.2:"run_T2.2_s203.npz",2.3:"run_T2.3_s101.npz",2.4:"run_T2.4_s204.npz"}
for T in sorted(files):
    d=np.load("/work/"+files[T]); m2b,m4b=d['m2b'],d['m4b']
    U,e=ujack(m2b,m4b)
    h=m2b.shape[0]//2
    u1=ujack(m2b[:h],m4b[:h])[0]; u2=ujack(m2b[h:],m4b[h:])[0]
    print(f"T={T} R={m2b.shape[1]} nblk={m2b.shape[0]} U={U:.6f} err={e:.6f} halves={u1:.5f},{u2:.5f}")