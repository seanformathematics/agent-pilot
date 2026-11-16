
import time,subprocess
time.sleep(400)
print(open('/work/prod.log').read())
print(subprocess.run('ls /work',shell=True,capture_output=True,text=True).stdout)
