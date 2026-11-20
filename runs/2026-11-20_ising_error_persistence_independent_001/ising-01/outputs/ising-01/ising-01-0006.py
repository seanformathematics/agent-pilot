
import subprocess
script = open('/work/run.py').read() if __import__('os').path.exists('/work/run.py') else None
print(script is not None)
