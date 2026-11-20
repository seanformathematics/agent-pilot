import sys, json; sys.path.insert(0,'/work')
import ising, statistics
temps=[1.9,2.0,2.1,2.2,2.3,2.4]
res=json.load(open('/work/results.json'))
rep={}
for k,T in enumerate(temps):
    vals=[res[str(T)]['U'] if str(T) in res else res[repr(T)]['U']]
    for j,seed in enumerate([7000+k, 9000+k]):
        n,m2,m4,ch=ising.simulate(T, seed, 30, init_up=(j==0))
        U,e,nt=ising.analyze(m2,m4,ch,200)
        vals.append(U)
        print(T, seed, "U=%.5f jk_err=%.5f n=%d"%(U,e,nt), flush=True)
    rep[T]=vals
    print("T",T,"all runs:",["%.5f"%v for v in vals],"mean=%.5f sd=%.5f"%(statistics.mean(vals),statistics.stdev(vals)), flush=True)
json.dump({str(k):v for k,v in rep.items()},open('/work/replicates.json','w'))
