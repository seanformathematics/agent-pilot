
import numpy as np
d = np.load('/work/prod1.npz')
m2=d['m2']; m4=d['m4']; am=d['am']; temps=d['temps']; R=int(d['R']); block=int(d['block']); equil=int(d['equil'])
nb, Rtot = m2.shape
n = nb*block
print("blocks",nb,"samples per replica",n)
def Ufun(S2,S4,n):
    return 1-(S4.mean()/n)/(3*(S2.mean()/n)**2)
out=[]
for k,T in enumerate(temps):
    sl=slice(k*R,(k+1)*R)
    S2=m2[:,sl].sum(axis=0); S4=m4[:,sl].sum(axis=0)
    U=Ufun(S2,S4,n)
    jk=np.array([Ufun(np.delete(S2,i),np.delete(S4,i),n) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    # split-half: first half of time blocks vs second half
    h=nb//2
    U1=Ufun(m2[:h,sl].sum(0),m4[:h,sl].sum(0),h*block)
    U2=Ufun(m2[h:,sl].sum(0),m4[h:,sl].sum(0),(nb-h)*block)
    # replica-level std of U_r for scale
    out.append((T,U,err))
    print(f"T={T:.2f} U={U:.5f} err={err:.5f} <|m|>={am[:,sl].sum()/(n*R):.4f}  halves: {U1:.4f} {U2:.4f}")
# drift check using time blocks of 100 blocks (10000 sweeps)
print("time-drift (U per 200-block chunk):")
for k in [0,4,7,9,11,13]:
    sl=slice(k*R,(k+1)*R)
    vals=[Ufun(m2[b:b+200,sl].sum(0),m4[b:b+200,sl].sum(0),200*block) for b in range(0,nb-199,200)]
    print(temps[k], np.round(vals,4))
