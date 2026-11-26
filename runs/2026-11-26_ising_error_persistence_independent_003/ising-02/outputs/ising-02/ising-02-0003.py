
import numpy as np, time, os, pickle
N=16
temps=[1.9,2.0,2.1,2.2,2.3,2.4]
R_per=1000
SEED=20240601
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c)[None] for c in (0,1)]
def sweep(s,beta,rng):
    p4=np.exp(-4*beta); p8=np.exp(-8*beta)
    for m in masks:
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nn
        r=rng.random(s.shape)
        acc=(dE<=0)|((dE==4)&(r<p4))|((dE==8)&(r<p8))
        acc&=m
        s[acc]*=-1
    return s
if not os.path.exists('/work/state.pkl'):
    rng=np.random.default_rng(SEED)
    Tv=np.repeat(np.array(temps),R_per)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(len(Tv),N,N))
    st=dict(s=s,Tv=Tv,rng=rng,done=0)
else:
    st=pickle.load(open('/work/state.pkl','rb'))
beta=(1.0/st['Tv']).reshape(-1,1,1)
t0=time.time()
nburn=3000
while st['done']<nburn and time.time()-t0<400:
    sweep(st['s'],beta,st['rng']); st['done']+=1
pickle.dump(st,open('/work/state.pkl','wb'))
print("sweeps done",st['done'],"time",time.time()-t0)
# quick look at U vs T with current config (no accumulation)
m=st['s'].reshape(len(st['Tv']),-1).mean(1)
for k,T in enumerate(temps):
    sel=np.arange(k*R_per,(k+1)*R_per)
    mm=m[sel]; m2=np.mean(mm**2); m4=np.mean(mm**4)
    print(T, "U inst", 1-m4/(3*m2*m2), "mean|m|",np.mean(np.abs(mm)))
