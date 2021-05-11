#! /usr/bin/env python
import pyrealsense2 as rs
from realsense_depth.realsense_depth import *
import matplotlib.pyplot as plt
from utils.dataset_processing import grasp, image
from utils.dataset_processing import evaluation1, grasp
from utils.dataset_processing.grasp import GraspRectangle
import math
import torch
import time
#from utils.dataset_processing import calibracion
from PIL import Image
from os.path import join
import numpy as np
import geometry_msgs.msg
import sys
import cv2
import scipy.ndimage as ndimage
import rospy
import std_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from models.common import post_process_output
from utils.timeit import TimeIt
import copy
#from helpers.transforms import *
#import pose_commander
#import tf.transformations as tft

device = torch.device('cpu')
crop_size=300
MODEL_FILE = 'ggcnn2_093'
rgbo = []
rgbfin = []
y_off=0
x_off=0
model = torch.load(MODEL_FILE, map_location='cpu')
rospy.init_node('save_img')
bridge = CvBridge()
cmd_pub = rospy.Publisher('ggcnn/rvalues', Float32MultiArray, queue_size=1)

rate = rospy.Rate(1) # ROS Rate at 5Hz
#iy=102
#ix=45
#Dy=196
#Dx=308
fx = 458.455478616934780
cx = 343.645038678435410
fy = 458.199272745572390
cy = 229.805975111304460
#fx = 585.6
cx = 320
#fy = 585.6
cy = 240

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def process_depth_image(depth, crop_size, out_size=crop_size, return_mask=False, crop_y_offset=0):
    imh, imw = depth.shape
    print(depth.shape)
    depth_crop = depth




    #(y, x) = np.where(out2 == 255)
    #(topy, topx) = (np.min(y), np.min(x))
    #(bottomy, bottomx) = (np.max(y), np.max(x))

    #depth_crop[:,0:topx-5] = 0.585
    #depth_crop[:,bottomx+5:] = 0.585
    #depth_crop[0:topy-5,:] = 0.585
    #depth_crop[bottomy+5:,:] = 0.585
    #depth_crop[out2 != 255] = 0.585



    #with TimeIt('1'):
       #depth_crop = depth[(imh - crop_size) // 2 + y_off:(imh - crop_size) // 2 + crop_size + y_off,
                           #(imw - crop_size) // 2+x_off:(imw - crop_size) // 2 + crop_size+x_off]



       #fig = plt.figure(figsize=(10, 10))
       #ax = fig.add_subplot(1, 1, 1)
       #ax.imshow(depth_crop, cmap='gray')

       #ax.set_title('Depth')
       #ax.axis('off')
       #plt.show()

    # Inpaint
    # OpenCV inpainting does weird things at the border.
    with TimeIt('2'):
        depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

    with TimeIt('3'):
        depth_crop[depth_nan_mask==1] = 0
        depth_crop[np.where(depth_crop == 0)]=0
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(depth_crop, cmap='gray')
        #for g in grasps:
        #    g.plot(ax)
        ax.set_title('Depth')
        ax.axis('off')
        plt.show()
    with TimeIt('4'):
        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        depth_scale = np.abs(depth_crop).max()
        depth_crop = depth_crop.astype(np.float32) / depth_scale  # Has to be float32, 64 not supported.



        with TimeIt('Inpainting'):
            depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        depth_crop = depth_crop[1:-1, 1:-1]
        depth_crop = depth_crop * depth_scale

    with TimeIt('5'):
        # Resize
        depth_crop = cv2.resize(depth_crop, (out_size, out_size), interpolation = cv2.INTER_AREA)

    if return_mask:
        with TimeIt('6'):
            depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
            depth_nan_mask = cv2.resize(depth_nan_mask, (out_size, out_size), interpolation = cv2.INTER_NEAREST)

        return depth_crop, depth_nan_mask
    else:
        return depth_crop

def predict(depth, process_depth=True, crop_size=crop_size, out_size=crop_size, depth_nan_mask=None, crop_y_offset=0, filters=(5.0, 3.0, 3.0)):
    if process_depth:
        depth, depth_nan_mask = process_depth_image(depth, crop_size, out_size=out_size, return_mask=True, crop_y_offset=0)

    # Inference
    depth = depth/1000
    #depth = depth/depth.max()
    depth = np.clip((depth - depth.mean()), -1, 1)
    #depth = cv2.blur(depth,(5,5))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(depth*100, cmap='gray')
    ax.set_title('Depthafterclip')
    ax.axis('off')


    depthn = depth.copy()
    #depthn[60:340,113:288] =depth[65,115]
    #depthn = ndimage.filters.gaussian_filter(depthn, 1)
    #depth = depth - depthn -depth[65,115]
    depthT = torch.from_numpy(depth.reshape(1, 1, out_size, out_size).astype(np.float32)).to(device)
    with torch.no_grad():
        pred_out = model(depthT)

    points_out = pred_out[0].cpu().numpy().squeeze()
    points_out[depth_nan_mask] = 0

    # Calculate the angle map.
    cos_out = pred_out[1].cpu().numpy().squeeze()
    sin_out = pred_out[2].cpu().numpy().squeeze()
    ang_out = np.arctan2(sin_out, cos_out) / 2.0

    width_out = pred_out[3].cpu().numpy().squeeze() * 150.0  # Scaled 0-150:0-1

    # Filter the outputs.

    points_out = ndimage.filters.gaussian_filter(points_out, filters[0])  # 3.0
    ang_out = ndimage.filters.gaussian_filter(ang_out, filters[1])
    width_out = ndimage.filters.gaussian_filter(width_out, filters[2])

    points_out = np.clip(points_out, 0.0, 1.0-1e-3)

    # SM
    # temp = 0.15
    # ep = np.exp(points_out / temp)
    # points_out = ep / ep.sum()

    # points_out = (points_out - points_out.min())/(points_out.max() - points_out.min())

    return points_out, ang_out, width_out, depth.squeeze()

def pushing(grasps, ix1, iy1, Dx1, Dy1, Dx, Dy):
    #find the objects that must be pushed first
    pushlist = np.zeros(len(grasps), dtype=int)
    cont =0
    for g in grasps:

        ang = g.angle
        l1x=g.length*np.cos(ang)*Dx/(2*crop_size)
        l2x=g.width*np.sin(ang)*Dx/(2*crop_size)
        l1y=g.length*np.sin(ang)*Dy/(2*crop_size)
        l2y=g.width*np.cos(ang)*Dy/(2*crop_size)
        mx = g.center[1]*Dx/crop_size-l1x-l2x
        max= g.center[1]*Dx/crop_size+l1x+l2x
        my= g.center[0]*Dy/crop_size-l1y-l2y
        may= g.center[0]*Dy/crop_size+l1y+l2y
        #print('mx, ix', mx, ix1)
        #print('mx, ix', mx, ix1)
        #print('max, ix+Dx', max, ix1+Dx1)
        #print('my, iy', my, iy1)
        #print('may, iy+Dy', may, iy1+Dy1)

        if my < iy1:
            pushlist[cont]=3
            g.length = 5
            g.width = 5
        if may > iy1+Dy1:
            pushlist[cont]=4
            g.length = 5
            g.width = 5
        if mx < ix1:
            pushlist[cont]=1
            g.length = 5
            g.width = 5
        if max > ix1+Dx1:
            pushlist[cont]=2
            g.length = 5
            g.width = 5
        cont =cont+1


    return pushlist, grasps

def graspdata(points_out, depthfin, grasps, ix, iy, Dx, Dy):


    #py,px=grasps[m].center
    #print("px,py: ",px,py)
    #ang = grasps[m].angle
    #print(ang)
    #width = grasps[m].width*2
    #print("width: ", width)
    #print('qf: ',points_out[py, px])
    #print('viejopixel: ', py, px)

    for g in grasps:
        #print('Dx y Dy:', Dx, Dy, ix, iy)
        #pyn = g.center[0]*Dy/crop_size + (480 - crop_size) // 2+y_off+iy
        #pxn = g.center[1]*Dx/crop_size + (640 - crop_size) // 2+x_off+ix
        pyn = g.center[0]*Dy/crop_size + iy+y_off
        pxn = g.center[1]*Dx/crop_size + ix+x_off
        ang=g.angle
        g.center = [pyn,pxn]
        l1 = g.length*np.cos(ang)*Dx/crop_size
        l2 = g.length*np.sin(ang)*Dy/crop_size
        l=np.sqrt(l1*l1+l2*l2)
        g.length =l
        g.width = l/2

            #pxn = np.round(pxn).astype(np.int)
            #pyn = np.round(pxn).astype(np.int)

    return grasps

def rvalues(grasp, depth, Dx, Dy, widthbinx,widthbiny):

    #print('length, width: ', grasp.length, grasp.width)
    #print('Nuevopixe    l: ', grasp.center[0],grasp.center[1])
    length = grasp.length
    width = grasp.width

    pyn=grasp.center[0]
    pxn=grasp.center[1]
    ang=grasp.angle
    point_depth = depth[pyn,pxn]
    x = (pxn - 318)/(fx)*point_depth
    y = (pyn - 229)/(fy)*point_depth
    #print('x1, y1' , x*100, y*100)
    #x = (pxn-cx)*0.66/(Dx+13)
    #y = (pyn-cy)*0.44/(Dy+25)
    x = (pxn-cx)*0.66/(widthbinx)
    y = (pyn-cy)*0.44/(widthbiny)

    z = point_depth
    x1 = (pxn+width*math.cos(ang)/2 - cx)/(fx)*point_depth
    y1 = (pyn+width*math.sin(ang)/2 - cy)/(fy)*point_depth
    x2 = (pxn-width*math.cos(ang)/2 - cx)/(fx)*point_depth
    y2 = (pyn-width*math.sin(ang)/2 - cy)/(fy)*point_depth

    rwidth =math.sqrt(math.pow((x1-x2),2)+math.pow((y1-y2),2))
    #print('x: ', x*100)
    #print('y: ', y*100)
    #print('z: ', z)
    #print('ang: ', ang*180/math.pi)
    #print('width: ', width)
    #print('rwidth: ', rwidth)


    return x,y,z, ang, rwidth



def find_pose():

        rospy.sleep(0.1)
        rgbo = rospy.wait_for_message('/camera/color/image_raw', Image)
        deptho = rospy.wait_for_message('/camera/depth/image_rect_raw', Image)
    #    rgbo = rospy.wait_for_message('/camera/rgb/image_color', Image)
        depthfin = bridge.imgmsg_to_cv2(deptho)
        rgbfin = bridge.imgmsg_to_cv2(rgbo)
        rgbfin1= cv2.cvtColor(rgbfin, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(depthfin, cmap='gray')
        #for g in grasps:
        #    g.plot(ax)
        ax.set_title('Depth')
        ax.axis('off')

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(rgbfin)
        ax.set_title('RGB')
        ax.axis('off')
        print(rgbfin.shape)
        print(depthfin.shape)
        plt.show()
        #img_p = np.average(rgbfin.astype(np.float64),axis=2)
        #print(img_p.shape)
        #img_p = np.tile(img_p[:,:,np.newaxis],(1,1,3))
        #img_p = np.tile(img_p,(1,1,3))

        #print(img_hc.shape)
        #print(np.amin(img_hc,axis=2).shape)
        #img_hc = k*(img_hc - np.tile(np.amin(img_hc,axis=2)[:,:,np.newaxis],(1,1,3))) # Manda el menor canal a 0 y multiplica
        #print(img_hc.dtype)

        #fig = plt.figure(figsize=(10, 10))
        #ax = fig.add_subplot(1, 2, 1)
        #plot = ax.imshow(img_hc)
        #ax.set_title('hc')
        #ax.axis('off')


        #ax = fig.add_subplot(1, 2, 2)
        #plot = ax.imshow(rgbfin)
        #ax.set_title('rgb')
        #ax.axis('off')
        #plt.show()

        #img_hc = cv2.cvtColor(img_hc.astype(np.uint8),cv2.COLOR_BGR2HSV)


        #fig = plt.figure(figsize=(10, 10))
        #ax = fig.add_subplot(1, 1, 1)
        #plot = ax.imshow(img_hc, cmap='hsv')
        #ax.set_title('hc')
        #ax.axis('off')
        #plt.show()

        #lim0 = [0, 0, 0]
        #lim1 = [120, 100, 100]
        #img_b = cv2.inRange(img_hc,lim0,lim1) #  Aqui ya deberias tener SOLO el perfil del canasto

        #iy, ix, Dy, Dx, widthbinx, widthbiny, iy1, ix1, Dy1, Dx1 = calibracion.calibracion(depthfin, rgbfin)


        #raw_input('done')
        #iy, ix, Dy, Dx, widthbinx, widthbiny, iy1, ix1, Dy1, Dx1 =[129, 155, 222, 329, 379, 252, 8, 6, 206, 317 ]

        #depthfin1 = depthfin[iy:Dy+iy, ix:Dx+ix]



        depthfin2 =np.float32(depthfin[20:460,40:600]/5)
        #nans, x= nan_helper(depthfin1)
        #depthfin1[nans]= np.interp(x(nans), x(~nans), depthfin1[~nans])
# threshold
        #mask = cv2.threshold(depthfin2,0,255,cv2.THRESH_BINARY)[1]

# erode mask to make black regions slightly larger
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        #mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)


# make mask 3 channel
        #mask = cv2.merge([mask,mask,mask])

# invert mask
        #mask_inv = 255 - mask

# get area of largest contour
        #contours = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #contours = contours[0] if len(contours) == 2 else contours[1]
        #perimeter_max = 0
        #for c in contours:
        #    perimeter = cv2.arcLength(c, True)
        #    if perimeter > perimeter_max:
        #        perimeter_max = perimeter


        #perimeter_max=5

        #radius = int(perimeter_max/2) + 1
        #if radius % 2 == 0:
        #    radius = radius + 1
            #print(radius)

# median filter input image
        #median = cv2.medianBlur(depthfin2, 5)

# apply mask to image
        #print(depthfin2.shape, mask.shape)
        #img_masked = cv2.bitwise_and(depthfin2, mask)

# apply inverse mask to median
        #median_masked = cv2.bitwise_and(median, mask_inv)

# add together
        #depthfin1 = cv2.add(img_masked,median_masked)




        depthfin1=depthfin2
        #depthfina = cv2.medianBlur(depthfin2, 3)
        #depthfinb = cv2.medianBlur(depthfina, 3)
        #depthfinc = cv2.medianBlur(depthfinb, 3)
        #depthfin1 = cv2.medianBlur(depthfinc, 3)
        #pipe = rs.pipeline()
        #cfg = rs.config()
        #cfg.enable_stream(640, 480);
        #pipe.start(cfg)
        #dec_filter = rs.decimation_filter ()   # Decimation - reduces depth frame density
        #spat_filter = rs.spatial_filter()          # Spatial    - edge-preserving spatial smoothing
        #temp_filter = rs.temporal_filter()    # Temporal   - reduces temporal noise
        #frames = pipe.wait_for_frames()
        #depth_frame = frames.get_depth_frame()
        #frames = pipe.wait_for_frames()
        #depth_frame = frames.get_depth_frame()

        #filtered = dec_filter.process(depth_frame)
        #filtered = spat_filter.process(filtered)
        #filtered = temp_filter.process(filtered)
        #depthfilt = bridge.imgmsg_to_cv2(filtered)

        #rgbfin1= cv2.cvtColor(rgbfin, cv2.COLOR_BGR2RGB)
        #cv2.imwrite('rgb.png', rgbfin1)
        #MODEL_FILE = 'training2_084'
        points_out, ang_out, width_out, depth = predict(depthfin1)
        grasps = grasp.detect_grasps(points_out, ang_out, 0.6, width_img=width_out, no_grasps=5)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(2, 2, 1)
        ax.imshow(depth, cmap='gray')
        for g in grasps:
            g.plot(ax)
        ax.set_title('Depth')
        ax.axis('off')

        ax = fig.add_subplot(2, 2, 2)
        plot = ax.imshow(points_out, cmap='jet', vmin=0, vmax=1)
        ax.set_title('quality')
        ax.axis('off')

        ax = fig.add_subplot(2, 2, 3)
        plot = ax.imshow(width_out, cmap='hsv', vmin=0, vmax=150)
        ax.set_title('width')
        ax.axis('off')

        ax = fig.add_subplot(2, 2, 4)
        plot = ax.imshow(ang_out, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
        ax.set_title('Angle')
        ax.axis('off')

        print(rgbfin.shape)
        print(depthfin.shape)
        plt.show()




while not rospy.is_shutdown():
    find_pose()

    rate.sleep()
