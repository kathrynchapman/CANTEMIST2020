import os

cmd = 'git add .'
os.popen(cmd).read()
cmd = 'git status'
os.popen(cmd).read()
inp = input("Message? Hit enter for simple 'bug fixes'\n\t")
if inp:
    message = inp
else:
    message = 'bug fixes'
cmd = 'git commit -m "{}"'.format(message)
os.popen(cmd).read()
cmd = 'git push origin master'
os.popen(cmd).read()