import cv2

img = cv2.imread('lena_gray_512.tif')
print(img)

cv2.imshow('test', img)
cv2.waitKey(0)
