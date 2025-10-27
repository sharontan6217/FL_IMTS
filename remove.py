import os
project_dir = os.getcwd()
print(project_dir)
for r,d,f in os.walk(project_dir):
    for f_ in f:
        print(f_)
        if '.' in f_ and f_.split('.')[1] in ['log','png']:
            print(r,f_)
            os.remove( r+'/'+f_)