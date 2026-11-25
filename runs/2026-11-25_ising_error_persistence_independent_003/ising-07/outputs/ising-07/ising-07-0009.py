
import numpy as np, glob
for f in sorted(glob.glob('/work/res_*.npz')):
    d=np.load(f); b2=d['blk_m2']; b4=d['blk_m4']
    n=b2.shape[0]; k=n//3
    def U(sl):
        m2=b2[sl].mean(0); m4=b4[sl].mean(0)
        return 1-m4.mean()/(3*m2.mean()**2)
    print(f.split('/')[-1], 'blocks',n, 'U first third %.5f  middle %.5f  last third %.5f'%(U(slice(0,k)),U(slice(k,2*k)),U(slice(2*k,n))))
