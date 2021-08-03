import open3d as o3d
import numpy as np
import json
import logging

def read_intrinsics(intrinsics_path):
    """
    Reads the json file with the camera parameteres and returns
    a dictionary with the json loaded and the intrinsics object

    Args:
        * intrinsics_path: path to the json file with the intrinsics
    """
    with open(intrinsics_path) as json_file:
        cam_params = json.load(json_file)

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        cam_params['depth intrinsics']['width'],
        cam_params['depth intrinsics']['height'],
        cam_params['depth intrinsics']['fx'],
        cam_params['depth intrinsics']['fy'],
        cam_params['depth intrinsics']['ppx'],
        cam_params['depth intrinsics']['ppy']
    )
    return cam_params, intrinsics

def load_rgbd_image(color_path, depth_path):
    """
    Loads RBDImage object given the paths to the color and depth images

    Args:
        * color_path: path to color image
        * depth_path: path to depth image
    """
    color = o3d.io.read_image(color_path)
    depth = o3d.io.read_image(depth_path)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False
    )
    return rgbd

def load_pcd(color_path, depth_path, intrinsics_path, d_scale=1, color_pcd=True):
    """
    Returns a PointCloud Object and a matrix representation of the PointCloud
    from the color and depth images.

    Args:
        * color_path: path to color image
        * depth_path: path to depth image
        * intrinsics_path: path to json file with camera parameters
        * d_scale: conversion factor to meters
    """
    # load params and images
    cam_params, _    = read_intrinsics(intrinsics_path)
    rgbd = load_rgbd_image(color_path, depth_path)
    depth_image = rgbd.depth

    # unpack params
    w = cam_params['depth intrinsics']['width']
    h = cam_params['depth intrinsics']['height']
    fx = cam_params['depth intrinsics']['fx']
    fy = cam_params['depth intrinsics']['fy']
    cx = cam_params['depth intrinsics']['ppx']
    cy = cam_params['depth intrinsics']['ppy']

    # axuxilary u and v arrays to compute the position of every pixel
    v = np.arange(h).reshape(-1,1)
    u = np.arange(w).reshape(1,-1)
    
    # compute channels
    z = np.asarray(depth_image) / d_scale
    x = (u-cx) * z/fx
    y = (v-cy) * z/fy
    
    # stack channels
    mat = np.stack((x, y, z), axis=-1)
    
    # map nan to 0
    mat = np.nan_to_num(mat)
    
    # PointCloud Object
    pcd = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(mat.reshape(-1,3))
    )
    # color image
    color_image = np.asarray(rgbd.color)
    
    # assign colors
    if color_pcd:
        pcd.colors = o3d.utility.Vector3dVector(color_image.reshape(-1,3)/255)
    
    # stack colors to get RGB-XYZ representation
    rgb_xyz = np.dstack((color_image, mat))

    return rgb_xyz, pcd
    
def draw_3d(geometries, flip=True, **kw_args):
    """
    Calls draw geometries o3d function after aplying transform so they
    wont be upside down in the visualizer. This function is intended to be
    used only with PointCloud and TriangleMesh objects.

    Args:
        * geometries: list of PointCloud objects
        * flip: boolean to indicate wheter or not to flip geometries
        * kw_args: keyword arguments to pass point of view
    """
    vis_list = []
    for g in geometries:
        # use copy constructor of the corresponding geometry type
        g_type = str(o3d.geometry.Geometry.get_geometry_type(g))
        if g_type == 'Type.TriangleMesh':
            vis_g = o3d.geometry.TriangleMesh(g)
        elif g_type == 'Type.PointCloud':
            vis_g = o3d.geometry.PointCloud(g)
        else:
            raise NotImplementedError(
                "Geometry type not implemented, please use regular o3d visualization"
            )
        # flip if needed
        if flip:
            vis_g.transform([
                [1, 0, 0, 0],
                [0,-1, 0, 0],
                [0, 0,-1, 0],
                [0, 0, 0, 1]   
            ])
        vis_list.append(vis_g)
    # draw_geometries call
    o3d.visualization.draw_geometries(vis_list, **kw_args)

def get_3d_box(obj, view):
    """
    Returns the bounding box used to segment a given object from a given camera view
    """
    # load json and fin cylinder
    with open("cylinders.json") as json_file:
        cylinders = json.load(json_file)
    for cylinder in cylinders:
        if obj in cylinder["objs"] and cylinder["view"] == view:
            break
    else:
        return None
    # create cylinder
    obj_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=cylinder["radius"], height=cylinder["height"]
    )
    # get rotation arround x axis matrix
    angle = cylinder["rotations"][0]["angle"] * np.pi/180
    c0, s0 = np.cos(angle), np.sin(angle)
    rot_x = np.array([
        [1, 0,  0],
        [0,c0,-s0],
        [0,s0, c0]
    ])
    # get rotation arround z axis if any and apply rotations
    if len(cylinder["rotations"]) == 1:
        obj_cylinder.rotate(rot_x)
    else:
        angle = cylinder["rotations"][1]["angle"] * np.pi/180
        c0, s0 = np.cos(angle), np.sin(angle)
        rot_z = np.array([
            [c0, -s0, 0],
            [s0,  c0, 0],
            [0,    0, 1]
        ])
        obj_cylinder.rotate(np.dot(rot_z, rot_x))
    # apply translation
    obj_cylinder.translate(cylinder["translation"])
    # return oriented bounding box
    return obj_cylinder.get_oriented_bounding_box()

def segment_object(pcd, bbox, obj, view, idx, eps=0.01, min_points=70, h=480, w=640):
    """
    Returns segmented object pointcloud and binary mask

    Args:
        * pcd: input PointCloud
        * bbox: OrientedBoundingBox objet to crop pcd
        * obj: class of the image being segmented, used for logging
        * view: camera point of view of the image being segmented, used for logging
        * idx: index of image set being processed, used for logging
        * eps: Optional, used in dbscan to define minimum distance to consider points part of the same cluster. Default to 0.01
        * min_points: Optional, used in dbscan to filter out small clusters as noise. Default to 70
    """
    log = logging.getLogger('')

    cropped = pcd.crop(bbox)
    log.debug(f"Cropped PointCloud for {obj}-{view}-{idx} image set: {str(cropped)}")

    log.debug(f"Running dbscan for {obj}-{view}-{idx} image set with eps={eps} min_points={min_points}")
    cluster_labels = np.array(
        cropped.cluster_dbscan(eps=eps, min_points=min_points)
    )
    n_clusters = cluster_labels.max() + 1
    log.debug(f"PointCloud for {obj}-{view}-{idx} image set has {n_clusters} cluster(s) inside the volume of interest")
    
    cluster_sizes = [len(np.where(cluster_labels==j)[0]) for j in np.unique(cluster_labels)]
    cluster_sizes = sorted(zip(cluster_sizes, np.unique(cluster_labels)), reverse=True, key=lambda x: x[0])
    log.debug(f"Sorted cluster sizes for {obj}-{view}-{idx} image set: {cluster_sizes}")

    filtered = cropped.select_by_index(np.where(cluster_labels>=0)[0])
    log.debug(f"Filtered PointCloud {obj}-{view}-{idx}: {str(filtered)}")

    dists = np.asarray(
        pcd.compute_point_cloud_distance(filtered)
    )

    inliers = np.where(dists==0)[0]
    segmented = pcd.select_by_index(inliers)

    mask = np.zeros((h*w,1), dtype=np.uint8)
    mask[inliers] = 255
    mask = mask.reshape((h,w))

    return segmented, mask
