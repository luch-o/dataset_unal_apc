from pcd_utils import load_pcd, segment_object, get_3d_box
import imageio
import os
import itertools
import logging
import logging.config
from logging_config import LOGGING_CONFIG

def process_image(obj, view, idx):
    """
    Function that process the image of a given object, view and index to obtain
    its binary mask and save it to disk

    Args:
        * obj: String with the object name
        * view: String, either 'front', 'mid', or 'height'
        * idx: String, idex number of the image to be processed
        * save: Boolean, if set to true saves mask to disk and returns None, 
            otherwise the mask is returned
    """
    intrinsics_path = os.path.join('img', obj, f'calib_data_{view}.json')
    color_path = os.path.join('img', obj, f'{view}_color_{idx}.jpg')
    depth_path = os.path.join('img', obj, f'{view}_depth_{idx}.png')
    mask_path = os.path.join('img', obj, f'{view}_mask_{idx}.png')

    _, pcd = load_pcd(color_path, depth_path, intrinsics_path)
    log.debug(f"PointCloud for {obj}-{view}-{idx}: {str(pcd)}")
    
    bbox = get_3d_box(obj, view)
    log.debug(f"OrientedBoundingBox for {obj}-{view}-{idx}: {str(bbox)}")

    _, mask = segment_object(pcd, bbox, obj, view, idx)

    imageio.imwrite(mask_path, mask)
    log.debug(f"Saving binary mask to {mask_path}")
    

if __name__ == "__main__":
    logging.config.dictConfig(LOGGING_CONFIG)
    log = logging.getLogger('')
    
    objects = os.listdir('img')
    #views = ('front', 'mid', 'high')
    views = ('mid',)
    objects = ("brownie",
            "cafe",
            "galletas",
            "desodorante_standing",
            "jugo",
            "aromaticas",
            "hueso",
            "jugo_standing",
            "esponja",
            "desodorante")

    log.info("Dataset images segmentation process started")
    try:
        for obj, view, idx in itertools.product(objects, views, range(200)):
            log.debug(f"Processing {obj}-{view}-{idx} set of images")
            process_image(obj, view, idx)
    except Exception:
        log.exception(f"Something failed while processing {obj}-{view}-{idx} set of images:")
    finally:
        log.info("Script execution finished")
    