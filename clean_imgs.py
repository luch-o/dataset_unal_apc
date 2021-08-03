"""
This script is intended to remove standing from filenames in standing objects
and to remove depth colored images
"""
import os

for obj in (obj for obj in os.listdir('img') if 'standing' in obj):
    for f in os.listdir(os.path.join('img', obj)):
        folder_path = os.path.join('img', obj)
        new_f = ''.join(f.split('_standing'))
        #print(f, os.path.join(folder_path, f), os.path.join(folder_path, new_f), sep='\n')
        #break
        os.rename(
            os.path.join(folder_path, f),
            os.path.join(folder_path, new_f)
        )