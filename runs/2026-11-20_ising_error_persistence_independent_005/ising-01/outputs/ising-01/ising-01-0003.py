import random, math, time, json, os
N=16; L=N*N
nb=[[((i//N+1)%N)*N+(i%N), ((i//N-1)%N)*N+(i%N), (i//N)*N+(i%N+1)%N, (i//N)*N+(i%N-1)%N] for i in range(L)]
def run(T, nupd, seed, nequil=20000, binsz=2000):
    rnd=random.Random(seed)
    s=[1]*L; M=L
    p=1-math.exp(-2.0/T)
    rr=rnd.random; ri=rnd.randrange
    bins=[]  # per bin: [n, sum M^2, sum M^4]
    cur=[0,0.0,0.0]
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
out={}
if os.path.exists('/work/res.json'): out=json.load(open('/work/res.json'))
t0=time.time()
for T in [1.9,2.0,2.1]:
    b=run(T,1000000,101)
    out[f"{T}_101"]=b
    json.dump(out,open('/work/res.json','w'))
    print(T,len(b),time.time()-t0,flush=True)