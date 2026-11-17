
import random, math, time
N=16
NN=N*N
nbr=[]
for i in range(N):
    for j in range(N):
        nbr.append([((i+1)%N)*N+j, ((i-1)%N)*N+j, i*N+(j+1)%N, i*N+(j-1)%N])

def run(T, nsteps, burn, seed, block=200):
    rng=random.Random(seed)
    p=1-math.exp(-2.0/T)
    spin=[1 if rng.random()<0.5 else -1 for _ in range(NN)]
    M=sum(spin)
    mark=[0]*NN
    stamp=0
    rnd=rng.random
    rint=rng.randrange
    blocks=[]  # (sum m2, sum m4, count)
    s2=0.0; s4=0.0; cnt=0
    tot=burn+nsteps
    for step in range(tot):
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

def U_from_blocks(blocks):
    S2=sum(b[0] for b in blocks); S4=sum(b[1] for b in blocks); C=sum(b[2] for b in blocks)
    m2=S2/C; m4=S4/C
    return 1-m4/(3*m2*m2)

def jack(blocks):
    n=len(blocks)
    U=U_from_blocks(blocks)
    Us=[]
    S2=sum(b[0] for b in blocks); S4=sum(b[1] for b in blocks); C=sum(b[2] for b in blocks)
    for b in blocks:
        m2=(S2-b[0])/(C-b[2]); m4=(S4-b[1])/(C-b[2])
        Us.append(1-m4/(3*m2*m2))
    mu=sum(Us)/n
    var=(n-1)/n*sum((u-mu)**2 for u in Us)
    return U, math.sqrt(var)

for T in [2.27,1.5]:
    t0=time.time()
    bl=run(T, 20000, 1000, 12345)
    U,e=jack(bl)
    print(T, U, e, len(bl), time.time()-t0, flush=True)
