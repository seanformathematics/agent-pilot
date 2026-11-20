
import json, numpy as np, math
d=json.load(open('/work/raw_T2.2.json'))
S2=np.array(d['S2']); S4=np.array(d['S4'])
N=16; NN=256; n=2000
def Uof(a2,a4,m):
    m2=a2/(m*NN*NN); m4=a4/(m*NN**4)
    return 1-m4/(3*m2*m2)
nbins=len(S2)
full=Uof(S2.sum(),S4.sum(),nbins*n)
print('full U',full)
tot2,tot4=S2.sum(),S4.sum()
jk=np.array([Uof(tot2-S2[b],tot4-S4[b],(nbins-1)*n) for b in range(nbins)])
err=math.sqrt((nbins-1)/nbins*np.sum((jk-jk.mean())**2))
print('jackknife err',err)
# group-size check
for g in [1,2,5,10,20,50]:
    nbg=nbins//g
    a2=S2[:nbg*g].reshape(nbg,g).sum(1); a4=S4[:nbg*g].reshape(nbg,g).sum(1)
    t2,t4=a2.sum(),a4.sum()
    jk=np.array([Uof(t2-a2[b],t4-a4[b],(nbg-1)*g*n) for b in range(nbg)])
    print('group',g,'err',math.sqrt((nbg-1)/nbg*np.sum((jk-jk.mean())**2)))
# per-bin m2 mean stats
m2b=S2/(n*NN*NN); print('<m2>',m2b.mean(), 'std of bin mean', m2b.std()/math.sqrt(nbins))
