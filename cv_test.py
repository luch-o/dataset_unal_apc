import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default='lena_gray_512.tif',
	help="path to input dataset of images")
ap.add_argument("-t", "--type", default="color",
    help="Image type, can be color, gray or depth. Must be lowercase.")
args = vars(ap.parse_args())

if args['type'] == 'color':
    flag = cv2.IMREAD_COLOR
elif args['type'] == 'gray':
    flag = cv2.IMREAD_GRAYSCALE
elif args['type'] ==  'depth':
    flag = cv2.IMREAD_ANYDEPTH
else:
    raise ValueError("Invalid image type specified")

img = cv2.imread(args['image'], flag)
print(img.shape)
print(img.dtype)

cv2.imshow('Image', img)
cv2.waitKey(0)
