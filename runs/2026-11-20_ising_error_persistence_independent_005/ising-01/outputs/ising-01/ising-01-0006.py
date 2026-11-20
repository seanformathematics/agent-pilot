import random, math, time, json
import numpy as np
N=16; L=N*N
nb=[[((i//N+1)%N)*N+(i%N), ((i//N-1)%N)*N+(i%N), (i//N)*N+(i%N+1)%N, (i//N)*N+(i%N-1)%N] for i in range(L)]
def run(T, nupd, seed, nequil=20000, binsz=2000):
    rnd=random.Random(seed)
    s=[1]*L; M=L
    p=1-math.exp(-2.0/T)
    rr=rnd.random; ri=rnd.randrange
    bins=[]; cur=[0,0.0,0.0]
    for it in range(nequil+nupd):
        i=ri(L); si=s[i]
        s[i]=-si; stack=[i]; csize=1
        while stack:
            j=stack.pop()
            for k in nb[j]:
                if s[k]==si and rr()<p:
                    s[k]=-si; stack.append(k); csize+=1
        M-=2*si*csize
        if it>=nequil:
            m2=M*M
            cur[0]+=1; cur[1]+=m2; cur[2]+=m2*m2
            if cur[0]==binsz:
                bins.append(cur); cur=[0,0.0,0.0]
    return bins
def U_of(B):
    n=B[:,0].sum(); m2=B[:,1].sum()/n; m4=B[:,2].sum()/n
    return 1-(m4/L**4)/(3*(m2/L**2)**2)
out=json.load(open('/work/res.json'))
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    b=run(T,500000,202); out[f"{T}_202"]=b
    B=np.array(b,float); n=len(B)
    jk=np.array([U_of(np.delete(B,i,axis=0)) for i in range(n)])
    print(T, round(U_of(B),5), round(float(np.sqrt((n-1)/n*np.sum((jk-jk.mean())**2))),5), flush=True)
json.dump(out,open('/work/res.json','w'))