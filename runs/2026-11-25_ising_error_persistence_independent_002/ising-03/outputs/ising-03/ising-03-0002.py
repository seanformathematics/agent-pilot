import shutil,subprocess
for c in ['gcc','cc','clang','tcc']:
    print(c, shutil.which(c))
