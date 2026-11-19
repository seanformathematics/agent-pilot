
import sys
sys.path.insert(0,'/work')
import numpy as np
from prod import *

Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
rows=[]
allz = {'d':[], 'i':[]}
print("T   | U_impr +- err_chain (err_block) | U_dir +- err_chain | (Ui-Ud) z | hot-cold z (i) | batch std / mean batch err (i) | batch z-rms | tau_d tau_i | Ubc-U")
for T in Ts:
    z = np.load("/work/prod_T%.1f.npz"%T)
    blocks = z['blocks']            # (K,nb,R,4)
    K,nb,R,_ = blocks.shape
    pooled = blocks.transpose(1,0,2,3).reshape(nb,K*R,4)   # (nb, K*R, 4); chain order = batch-major
    ai = analyze(pooled,'i'); ad = analyze(pooled,'d')
    # difference estimator (improved - direct): jackknife over chains
    chain = pooled.mean(axis=0)  # (K*R,4)
    n = chain.shape[0]; tot = chain.sum(axis=0)
    loo = (tot[None,:]-chain)/(n-1)
    Ui_loo = Ufun(loo[:,2],loo[:,3]); Ud_loo = Ufun(loo[:,0],loo[:,1])
    dif = Ui_loo-Ud_loo
    dif_err = np.sqrt((n-1)/n*((dif-dif.mean())**2).sum())
    dif_val = ai['U']-ad['U']
    # per-batch results
    Ub = {}; Eb={}
    for w in ('d','i'):
        us=[];es=[]
        for k in range(K):
            a = analyze(blocks[k],w); us.append(a['U']); es.append(a['err_chain'])
        Ub[w]=np.array(us); Eb[w]=np.array(es)
        allz[w].extend(list((Ub[w]-ai['U' if w=='i' else 'U'] if False else Ub[w]-(ai['U'] if w=='i' else ad['U']))/Eb[w]))
    zrms = np.sqrt(np.mean(((Ub['i']-ai['U'])/Eb['i'])**2))
    td,_ = tau_int(z['acf_d'].mean(axis=0)); ti,_ = tau_int(z['acf_i'].mean(axis=0))
    hc = (ai['U_hot']-ai['U_cold'])/np.hypot(ai['err_hot'],ai['err_cold'])
    print("%.1f | %.5f +- %.5f (%.5f) | %.5f +- %.5f | %.2f | %.2f | %.5f / %.5f | %.2f | %.2f %.2f | %.1e"%(
        T, ai['U'], ai['err_chain'], ai['err_block'], ad['U'], ad['err_chain'], dif_val/dif_err, hc,
        Ub['i'].std(ddof=1), Eb['i'].mean(), zrms, td, ti, ai['U_biascorr']-ai['U']))
print()
for w in ('d','i'):
    zz=np.array(allz[w]); print("estimator %s: batch z-scores over all T (n=%d): mean %.3f, rms %.3f"%(w,len(zz),zz.mean(),np.sqrt((zz**2).mean())))
