
import random, math, time, json, os
N=16
NN=N*N
nbr=[]
for i in range(N):
    for j in range(N):
        nbr.append([((i+1)%N)*N+j, ((i-1)%N)*N+j, i*N+(j+1)%N, i*N+(j-1)%N])

def run(T, nsteps, burn, seed, block):
    rng=random.Random(seed)
    p=1-math.exp(-2.0/T)
    spin=[1 if rng.random()<0.5 else -1 for _ in range(NN)]
    M=sum(spin)
    mark=[0]*NN
    stamp=0
    rnd=rng.random
    rint=rng.randrange
    blocks=[]
    s2=0.0; s4=0.0; cnt=0
    for step in range(burn+nsteps):
        seed_site=rint(NN)
        s0=spin[seed_site]
        stamp+=1
        mark[seed_site]=stamp
        stack=[seed_site]
        cl=[seed_site]
        while stack:
            x=stack.pop()
            for y in nbr[x]:
                if mark[y]!=stamp and spin[y]==s0 and rnd()<p:
                    mark[y]=stamp
                    stack.append(y)
                    cl.append(y)
        for x in cl:
            spin[x]=-s0
        M-=2*s0*len(cl)
        if step>=burn:
            m2=(M/NN)**2
            s2+=m2; s4+=m2*m2; cnt+=1
            if cnt==block:
                blocks.append((s2,s4,cnt)); s2=0.0; s4=0.0; cnt=0
    return blocks

def jack(blocks):
    n=len(blocks)
    S2=sum(b[0] for b in blocks); S4=sum(b[1] for b in blocks); C=sum(b[2] for b in blocks)
    m2=S2/C; m4=S4/C
    U=1-m4/(3*m2*m2)
    Us=[]
    for b in blocks:
        a2=(S2-b[0])/(C-b[2]); a4=(S4-b[1])/(C-b[2])
        Us.append(1-a4/(3*a2*a2))
    mu=sum(Us)/n
    var=(n-1)/n*sum((u-mu)**2 for u in Us)
    return U, math.sqrt(var), n

Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
prog={}
if os.path.exists('/work/progress.json'):
    prog=json.load(open('/work/progress.json'))
t0=time.time()
for iT,T in enumerate(Ts):
    key=str(T)
    if key in prog: continue
    if time.time()-t0>420: break
    chains=[]
    for c in range(8):
        chains.append(run(T, 30000, 2000, 7000+100*iT+c, 500))
    allb=[b for r in chains for b in r]
    U,e,n=jack(allb)
    prog[key]=dict(U=U,err=e,nblocks=n,chainU=[jack(r)[0] for r in chains],seeds=[7000+100*iT+c for c in range(8)])
    json.dump(prog, open('/work/progress.json','w'))
    print(T, round(U,6), round(e,6), n, round(time.time()-t0,1), flush=True)
print('done' if len(prog)==len(Ts) else 'partial', len(prog))
