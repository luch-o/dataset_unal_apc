# Code downloaded from pysource and adapted
# https://pysource.com/2021/03/11/distance-detection-with-depth-camera-intel-realsense-d435i/

import pyrealsense2 as rs
import numpy as np

# constants
WIDTH = 640
HEIGHT = 480
FRAME_RATE = 30
TEMP_FILT_FRAMES = 5

class DepthCamera:
    """
    Custom class to encapsulate realsense camera frame obtention
    """
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # enable stream
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,  FRAME_RATE)
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FRAME_RATE)

        # create align and colorizer objects
        self.colorizer = rs.colorizer()
        self.align = rs.align(rs.stream.color)

        # create post-processing depth filters
        self.decimation = rs.decimation_filter()
        self.depth_to_disparity = rs.disparity_transform(True)
        self.spatial = rs.spatial_filter()
        # enable spactial filter to perform some basic hole filling
        self.spatial.set_option(rs.option.holes_fill, 3)
        self.temporal = rs.temporal_filter()
        self.disparity_to_depth = rs.disparity_transform(False)
        self.hole_filling = rs.hole_filling_filter()        

        # Start streaming
        self.pipeline.start(config)

        # Skip 5 first frames to give the Auto-Exposure time to adjust
        for _ in range(5):
            self.pipeline.wait_for_frames()


    def get_calibration_params(self):
        """
        get a nested dictionary json-like with calibration intrinsics from color and depth, and
        extrinsics from depth to color 
        """
        profile = self.pipeline.get_active_profile()

        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()

        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()

        depth_to_color_extrinsics = depth_profile.get_extrinsics_to(color_profile)
        
        calibration_params = {'color intrinsics': color_intrinsics,
                              'depth intrinsics': depth_intrinsics,
                              'depth to color extrinsics': depth_to_color_extrinsics}

        calibration_data = {}
        for name, params in calibration_params.items():
            calibration_data[name] = {attr:str(getattr(params, attr)) if attr == 'model' else getattr(params, attr)
                                      for attr in dir(params) if not attr.startswith('__')}
        
        return calibration_data


    def get_frame(self, align=True, apply_filters=False):
        """
        If succesful frame retrieval, return True and color,
        depth and colorized with jet colormap depth images;
        otherwise returns False, and None instead of each image.        
        """

        # capture a frameset or multiple framesets if apply filters is set to True
        for _ in range(TEMP_FILT_FRAMES):
            frameset = self.pipeline.wait_for_frames()
            # if align is set to true, use align objecto to align frames
            if align:
                frameset = self.align.process(frameset)
            depth_frame = frameset.get_depth_frame()

            # break after taking first frameset if apply_filters is False
            if not apply_filters: break

            # apply filters
            #depth_frame = self.decimation.process(depth_frame)
            depth_frame = self.depth_to_disparity.process(depth_frame)
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.temporal.process(depth_frame)
            depth_frame = self.disparity_to_depth.process(depth_frame)
            depth_frame = self.hole_filling.process(depth_frame)
        
        # get color frame and colorize depth frame
        color_frame = frameset.get_color_frame()
        depth_colored_frame = self.colorizer.colorize(depth_frame)
        
        # get numpy arrays from frame objects
        depth_colored_image = np.asanyarray(depth_colored_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # return images as numpy arrays
        if not depth_frame or not color_frame:
            return False, None, None, None
        return True, color_image, depth_image, depth_colored_image


    def release(self):
        """
        stop realsense pipeline
        """
        self.pipeline.stop()
