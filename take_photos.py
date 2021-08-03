
from turntable import Turntable
from realsense_depth.realsense_depth import DepthCamera
import cv2
import argparse
import os
import json
import time

# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--class", type=str, required=True,
	help="class name of the object")
ap.add_argument("-p", "--prefix", type=str, required=True,
	help="prefix indicating the camera angle. Must be front, mid or high")
ap.add_argument("-o", "--orientation", type=bool, default=False,
    help="orientation of the object in the photos" + 
         "should be set to True if the object is standing." +  
         "Set to False by default.")
ap.add_argument("-t", "--time", type=bool, default=False,
    help="Whether to print execution time. Must be a boolean")    
args = vars(ap.parse_args())

# check prefix
if args['prefix'] in {'front', 'mid', 'high'}:
    prefix = args['prefix'] + ('_standing' if args['orientation'] else '')
else:    
    raise ValueError("Prefix must be front, mid or high")

# Make new directory to store the images
path = os.path.join('img', args['class'])
if not os.path.exists(path):
    os.mkdir(path)
os.chdir(path)

# initialize turntable and camera
dc = DepthCamera()
my_turntable = Turntable()

start = time.time()
# write calibraton data to a json file
with open(f"calib_data_{prefix}.json", 'w+') as calib_file:
    json.dump(dc.get_calibration_params(), calib_file, indent=2)

try:
    # Take 200 set of frames, turning the table  16 steps at a time (1.8 degrees)
    for i in range(200):
        print("Taking set of frames", i+1)
        # retrieve and store frames    
        _, color_img, depth_img, depth_colored_img = dc.get_frame(apply_filters=True)
        cv2.imwrite(f'{prefix}_color_{i}.jpg', color_img)
        cv2.imwrite(f'{prefix}_depth_{i}.tiff', depth_img)
        cv2.imwrite(f'{prefix}_depth_jet_{i}.jpg', depth_colored_img)
        # move turntable
        my_turntable.move_table(16)

except KeyboardInterrupt:
    pass    

exec = time.time() - start
if args['time']:
    print(f"Execution time: {exec//60:.0f}:{int(exec%60):0>2}")
    
print("Finished, releasing camera and closing serial port")
dc.release()
my_turntable.close()
