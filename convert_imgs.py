# This script is meant to convert .tiff depth images to 16 bit png images
# it will take a while running but only will be run once

import imageio
import os

# for each folder in img
for obj in os.listdir('img'):
    dir_path = os.path.join('img', obj)

    # get file paths with .tiff extension
    depth_imgs = [img for img in os.listdir(dir_path) if img.endswith('.tiff')]
    if not depth_imgs:
        print('No tiff files found in', obj)
        continue

    # for each tiff img
    for depth_img in depth_imgs:
        # keep track of complete old path and new path
        old_path = os.path.join(dir_path, depth_img)
        new_path = os.path.splitext(old_path)[0] + '.png'

        # load image from old path and save to new path
        img = imageio.imread(old_path)
        print('writing', new_path)
        imageio.imwrite(new_path, img)

        # delete new path
        print('removing', old_path)
        os.remove(old_path)
