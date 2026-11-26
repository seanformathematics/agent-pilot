
import numpy as np, time, os, pickle
N=16
temps=[1.9,2.0,2.1,2.2,2.3,2.4]
R_per=1000
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
st=pickle.load(open('/work/state.pkl','rb'))
beta=(1.0/st['Tv']).reshape(-1,1,1)
BL=50
blocks_m2=[]; blocks_m4=[]
if os.path.exists('/work/meas.npz'):
    d=np.load('/work/meas.npz'); blocks_m2=list(d['m2']); blocks_m4=list(d['m4'])
t0=time.time()
R=len(st['Tv'])
while st.get('mdone',0)<10000 and time.time()-t0<330:
    acc2=np.zeros(R); acc4=np.zeros(R)
    for _ in range(BL):
        sweep(st['s'],beta,st['rng']); st['done']+=1
        m=st['s'].reshape(R,-1).mean(1)
        m2=m*m
        acc2+=m2; acc4+=m2*m2
    st['mdone']=st.get('mdone',0)+BL
    blocks_m2.append(acc2/BL); blocks_m4.append(acc4/BL)
pickle.dump(st,open('/work/state.pkl','wb'))
np.savez('/work/meas.npz',m2=np.array(blocks_m2),m4=np.array(blocks_m4))
print("measure sweeps",st['mdone'],"total",st['done'],"time",time.time()-t0)
