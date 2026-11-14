
exec(open('/tmp/run_python/'+__import__('os').listdir('/tmp/run_python')[-1]).read().split("Ts=[")[0]) if False else None
import numpy as np, random, math, json
N=16;L=N*N
nb=[[((i//N+1)%N)*N+i%N,((i//N-1)%N)*N+i%N,(i//N)*N+(i+1)%N,(i//N)*N+(i-1)%N] for i in range(L)]
def run(T,seed,nth,nmeas):
    rnd=random.Random(seed);p=1-math.exp(-2/T)
    s=[rnd.choice((-1,1)) for _ in range(L)]
    M=sum(s);m2=0.;m4=0.;rnd_=rnd.random
    for it in range(nth+nmeas):
        i=rnd.randrange(L);sp=s[i];s[i]=-sp;st=[i];c=1
        while st:
            j=st.pop()
            for k in nb[j]:
                if s[k]==sp and rnd_()<p:
                    s[k]=-sp;st.append(k);c+=1
        M-=2*sp*c
        if it>=nth:
            m=M/L;a=m*m;m2+=a;m4+=a*a
    return 1-(m4/nmeas)/(3*(m2/nmeas)**2)
r=json.load(open('/work/r.json'))
for T in [2.5,2.6,2.7,2.8]:
    us=[run(T,7000+int(T*10)*100+k,1000,100000) for k in range(10)]
    r[str(T)]=(np.mean(us),np.std(us,ddof=1)/np.sqrt(10));print(T,r[str(T)],flush=True)
json.dump(r,open('/work/r2.json','w'))
