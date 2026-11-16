import os,glob
for p in glob.glob('/proc/[0-9]*/cmdline'):
    try:
        c=open(p,'rb').read().replace(b'\0',b' ').decode()
        if 'sim' in c: print(p,c)
    except Exception as e: pass
print(os.listdir('/work'), open('/work/sim.log').read())