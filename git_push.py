import os

cmd = 'git add .'
os.popen(cmd).read()
cmd = 'git status'
os.popen(cmd).read()
cmd = 'git commit -m "bug fixes"'
os.popen(cmd).read()
cmd = 'git push origin master'
os.popen(cmd).read()