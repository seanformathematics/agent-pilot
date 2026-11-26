
import numpy as np, pickle, time
d=pickle.load(open('/work/all.pkl','rb'))
b2=d['b2']; b4=d['b4']
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]; R=1000
per=b2.sum(0)/(b2.shape[0]*100)  # mean m^2 per replica
for k,T in enumerate(Ts):
    p=per[k*R:(k+1)*R]
    print(T, 'min meanm2',p.min().round(4),'frac<0.3',(p<0.3).mean(), 'median',np.median(p).round(4))
# independent check from ordered start
N=16
Tsel=[1.9,2.0,2.1]
R2=500
Tarr=np.repeat(np.array(Tsel),R2)
beta=(1.0/Tarr).astype(np.float32).reshape(-1,1,1)
rng=np.random.default_rng(777)
S=np.ones((len(Tarr),N,N),dtype=np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(S):
    for m in masks:
        nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.float32)
        dE=2*S*nn
        acc=(dE<=0)|(rng.random(S.shape,dtype=np.float32)<np.exp(-beta*dE))
        S[acc&m]*=-1
    return S
t0=time.time()
for _ in range(1500): sweep(S)
s2=np.zeros(len(Tarr));s4=np.zeros(len(Tarr));n=0
for _ in range(3000):
    sweep(S); m=np.abs(S.reshape(len(Tarr),-1).mean(1)); s2+=m**2; s4+=m**4; n+=1
for k,T in enumerate(Tsel):
    sl=slice(k*R2,(k+1)*R2)
    M2=s2[sl].sum()/(n*R2); M4=s4[sl].sum()/(n*R2)
    print('ordered-start check T',T,'U=',1-M4/(3*M2**2), time.time()-t0)
