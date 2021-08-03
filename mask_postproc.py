import cv2
import os
from itertools import product
import logging
import logging.config
from logging_config import LOGGING_CONFIG_POSTPROC

kernel_size = (7,7)
kernel_shapes = {
    cv2.MORPH_RECT: "Rectangle"
}

def morph_filter(mask, kernel):
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

def main():
    logging.config.dictConfig(LOGGING_CONFIG_POSTPROC)
    log = logging.getLogger('')

    log.info("Starting mask postrprocessing script")

    objects = os.listdir('img')
    views = ('front', 'mid', 'high')

    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    log.info(f"Using {kernel_shapes[shape]} structuring element of size {kernel_size}")
    
    try:
        for obj, view, idx in product(objects, views, range(200)):
            # load mask
            mask_path = os.path.join('img', obj, f'{view}_mask_{idx}.png')
            log.debug(f"Processing {mask_path}")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # process mask
            filtered = morph_filter(mask, kernel)
            # save to disk
            save_path = os.path.join('img', obj, f'{view}_mask_proc_{idx}.png')
            log.debug(f"Saving {save_path} to disk")
            cv2.imwrite(save_path, filtered)
            
    except Exception as e:
        log.exception(f"Something failed while processing {mask_path}")
    finally: 
        log.info("Script execution finished")        

if __name__ == "__main__":
    main()
