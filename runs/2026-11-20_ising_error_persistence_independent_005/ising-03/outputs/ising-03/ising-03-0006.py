
import sys; sys.path.insert(0,'/work')
import sim, json, time
for T,seed in [(2.1,2103),(2.3,2303),(2.4,2404)]:
    t0=time.time()
    S2,S4=sim.simulate(T,seed,2_000_000,20000,2000)
    U,err=sim.analyse(S2,S4,2000)
    print('T',T,'U',U,'err',err,'time',time.time()-t0, flush=True)
    json.dump({'T':T,'seed':seed,'S2':S2.tolist(),'S4':S4.tolist()},open(f'/work/raw_T{T}.json','w'))
