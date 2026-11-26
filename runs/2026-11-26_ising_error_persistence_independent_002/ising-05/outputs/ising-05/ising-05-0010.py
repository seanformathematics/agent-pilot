
import numpy as np, math
L=4; T=2.4
# enumerate
idx=np.arange(2**(L*L))
bits=((idx[:,None]>>np.arange(L*L))&1)*2-1
S=bits.reshape(-1,L,L)
E=-(np.sum(S*np.roll(S,1,1),axis=(1,2))+np.sum(S*np.roll(S,1,2),axis=(1,2)))
w=np.exp(-(E-E.min())/T); w/=w.sum()
m=S.reshape(-1,L*L).mean(1)
print("exact <E>/N", np.sum(w*E)/(L*L), "<m^2>", np.sum(w*m*m), "<|m|>", np.sum(w*np.abs(m)))
# reference metropolis
rng=np.random.default_rng(5)
s=rng.choice([-1,1],size=(L,L))
Es=[];ms=[]
for sw in range(200000):
    i,j=rng.integers(L,size=2)
    h=s[(i+1)%L,j]+s[(i-1)%L,j]+s[i,(j+1)%L]+s[i,(j-1)%L]
    dE=2*s[i,j]*h
    if dE<=0 or rng.random()<math.exp(-dE/T): s[i,j]=-s[i,j]
    if sw>20000 and sw%5==0:
        mm=s.mean(); ms.append(mm*mm); Es.append(-(np.sum(s*np.roll(s,1,0))+np.sum(s*np.roll(s,1,1)))/(L*L))
print("single-spin MC <E>/N", np.mean(Es), "<m^2>", np.mean(ms))
