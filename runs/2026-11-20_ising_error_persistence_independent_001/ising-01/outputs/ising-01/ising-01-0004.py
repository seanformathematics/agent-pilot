
import time,glob,subprocess
time.sleep(300)
print(open('/work/log.txt').read())
print(glob.glob('/work/*.npz'))
print(subprocess.run('ls -la /work',shell=True,capture_output=True,text=True).stdout)
