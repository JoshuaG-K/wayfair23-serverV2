import json
import re
import copy
from scipy.spatial.transform import Rotation
import numpy as np
import open3d as o3d
import argparse
import math
from typing import List, Literal, Optional, Tuple
# import torch
# from jaxtyping import Float
# from numpy.typing import NDArray
# from torch import Tensor


APP_TRANSFORM_MATRIX  = 'transform_matrix'
TRANSFORM_MATRIX = 'transform_matrix'
APP_IMG_PATH = 'file_path'
MODEL_POSITION_MATRIX = 'position'
MODEL_ROTATION_MATRIX = 'rotation'
MODEL_IMG_PATH = 'img_name'

def get_bounding_box(bounding_box_file_path):
    # Open the .json file for reading
    with open(bounding_box_file_path, 'r') as file:
        # Load the data from the file
        bbox_json_data = json.load(file)

    bbox_width = (bbox_json_data['positions']['bot_left_back']['x'] - bbox_json_data['positions']['bot_right_back']['x'])**2 + (bbox_json_data['positions']['bot_left_back']['z'] - bbox_json_data['positions']['bot_right_back']['z'])**2
    bbox_height = abs(bbox_json_data['positions']['bot_right_back']['y'] - bbox_json_data['positions']['top_right_back']['y'])
    bbox_depth = (bbox_json_data['positions']['bot_right_back']['z'] - bbox_json_data['positions']['bot_right_front']['z'])**2 + (bbox_json_data['positions']['bot_right_back']['x'] - bbox_json_data['positions']['bot_right_front']['x'])**2
    bbox_center_x = bbox_json_data['center']['x']
    bbox_center_y = bbox_json_data['center']['y']
    bbox_center_z = bbox_json_data['center']['z']

    bbox = [bbox_center_x, bbox_center_y, bbox_center_z, bbox_width, bbox_height, bbox_depth]

    return bbox

def get_bounding_box_from_dict(bbox_json_data):

    bbox_width = (bbox_json_data['positions']['bot_left_back']['x'] - bbox_json_data['positions']['bot_right_back']['x'])**2 + (bbox_json_data['positions']['bot_left_back']['z'] - bbox_json_data['positions']['bot_right_back']['z'])**2
    bbox_height = abs(bbox_json_data['positions']['bot_right_back']['y'] - bbox_json_data['positions']['top_right_back']['y'])
    bbox_depth = (bbox_json_data['positions']['bot_right_back']['z'] - bbox_json_data['positions']['bot_right_front']['z'])**2 + (bbox_json_data['positions']['bot_right_back']['x'] - bbox_json_data['positions']['bot_right_front']['x'])**2
    bbox_center_x = bbox_json_data['center']['x']
    bbox_center_y = bbox_json_data['center']['y']
    bbox_center_z = bbox_json_data['center']['z']

    bbox = [bbox_center_x, bbox_center_y, bbox_center_z, bbox_width, bbox_height, bbox_depth]

    return bbox

# Code adapted from Chat-GPT
def create_snaking_line(bbox, num_points, num_turns=3, radius_scale=4):
    # Generate a snaking line on the ellipsoid surface (helix-like path)
    t = np.linspace(0, num_turns * 2 * np.pi, num_points)
    x = bbox[0] + radius_scale * bbox[3] / 2 * np.cos(t)
    y = bbox[1] + bbox[4] * t / (num_turns * 2 * np.pi) - bbox[4]/2
    z = bbox[2] + radius_scale * bbox[5] / 2 * np.sin(t)

    return x, y, z

# Code gotten from chat-gpt
def find_closest_point_on_ellipsoid(camera_position, bbox):
    u = np.arctan2(camera_position[1] - bbox[1], camera_position[0] - bbox[0])
    v = np.arctan2(np.sqrt((camera_position[0] - bbox[0])**2 + (camera_position[1] - bbox[1])**2), camera_position[2] - bbox[2])

    x = bbox[0] + bbox[3] / 2 * np.cos(u) * np.sin(v)
    y = bbox[1] + bbox[4] / 2 * np.sin(u) * np.sin(v)
    z = bbox[2] + bbox[5] / 2 * np.cos(v)

    return np.array([x, y, z])

# Code adapted from chat-gpt
def get_rotation_angles(point1, point2):
    # Does not try to even calculate a roll rotation
    # Calculate the vector from the camera position to the closest point
    camera_to_closest_vector = np.array(point1) - np.array(point2)

    # Normalize the vector
    camera_to_closest_vector_normalized = camera_to_closest_vector / np.linalg.norm(camera_to_closest_vector)

    # Calculate pitch and yaw angles
    pitch = np.arcsin(-camera_to_closest_vector_normalized[2])
    yaw = np.arctan2(camera_to_closest_vector_normalized[1], camera_to_closest_vector_normalized[0])

    return pitch, yaw

# Function gotten from chat-gpt
def create_camera_pose_matrix(x, y, z, pitch, yaw):
    pitch *= -1
    yaw *= -1
    # Translation matrix
    translation_matrix = np.array([[1, 0, 0, x],
                                   [0, 1, 0, y],
                                   [0, 0, 1, z],
                                   [0, 0, 0, 1]])
    
    # rotation_y = np.array([[np.cos(pitch), 0, np.sin(pitch), 0],
    #                        [0, 1, 0, 0],
    #                        [-np.sin(pitch), 0, np.cos(pitch), 0],
    #                        [0, 0, 0, 1]])

    # rotation_z = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
    #                        [np.sin(yaw), np.cos(yaw), 0, 0],
    #                        [0, 0, 1, 0],
    #                        [0, 0, 0, 1]])

    # Rotation matrices
    rotation_x = np.array([[1, 0, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch), 0],
                    [0, np.sin(pitch), np.cos(pitch), 0],
                    [0, 0, 0, 1]])
    
    rotation_y = np.array([[np.cos(yaw), 0, np.sin(yaw), 0],
                           [0, 1, 0, 0],
                           [-np.sin(yaw), 0, np.cos(yaw), 0],
                           [0, 0, 0, 1]])

    # Combine translation and rotation

    # # Using y and z (if x is forward)
    # pose_matrix = np.dot(np.dot(translation_matrix, rotation_z), rotation_y)

    # Using x and y (if z is forward)
    pose_matrix = np.dot(np.dot(translation_matrix, rotation_x), rotation_y) 

    return pose_matrix

# Function takes in a camera position and a bounding box and outputs the camera 6DoF coordinates in a 4x4 matrix for the camera to look at the closest point to the ellipsoid 
# that is defined by the bounding box
def get_camera_matrix(camera_position, bbox):
    closest_point = find_closest_point_on_ellipsoid(camera_position, bbox)
    center = np.array([bbox[0], bbox[1], bbox[2]])
    rotation_angles = get_rotation_angles(center, camera_position)
    camera_pose_matrix = create_camera_pose_matrix(camera_position[0], camera_position[1], camera_position[2], rotation_angles[0], rotation_angles[1])

    return camera_pose_matrix


"""
Given a bounding box, the below code creates an ellipse around the bounding box. It takes in a 
function to draw a path around this ellipse which the camera poses will be sampled from. The
cameras will face the closest point on the ellipse which is how it gets the angle.
"""
def get_camera_poses(bbox, shape_func, num_points):
    camera_positions_x, camera_positions_y, camera_positions_z = shape_func(bbox, num_points)
    camera_poses = np.ones([len(camera_positions_x), 4, 4])
    # This part can be optimized by making more use of np.arrays
    for i in range(len(camera_positions_x)):
        camera_position = np.array([camera_positions_x[i], camera_positions_y[i], camera_positions_z[i]])
        camera_pose_matrix = get_camera_matrix(camera_position, bbox)
        camera_poses[i] = camera_pose_matrix
    return camera_poses

def write_json(json_info, output_path="output.json"):
    # Write the dictionary to the JSON file
    with open(output_path, 'w') as json_file:
        json.dump(json_info, json_file)

    print(f"The dictionary has been written to {output_path} as a JSON object.")

def extract_poses_from_app_json(app_cameras_file_path):
    """
    This function assumes that images/0 is at the 0th location in the frames list 

    The way we parse the json is from this json format for transforms.json https://drive.google.com/drive/folders/1rbmKD0MjbBhTPawfZXfPey5b-ldVpk8s 
    """
    # Open the .json file for reading
    with open(app_cameras_file_path, 'r') as file:
        # Load the data from the file
        app_data = json.load(file)
    num_frames = len(app_data['frames'])
    app_camera_poses = []
    app_camera_poses_dict = {}

    pattern = ".*/(\d+)$"

    for i in range(num_frames):
        frame = app_data['frames'][i]
        frame_id = frame[APP_IMG_PATH]
        frame_id = re.findall(pattern, frame_id, re.IGNORECASE)
        frame_id = frame_id[0]

        transformation_matrix = np.array(frame[APP_TRANSFORM_MATRIX])
        app_camera_poses.append(transformation_matrix)
        app_camera_poses_dict[frame_id] = transformation_matrix
    app_camera_poses = np.array(app_camera_poses)
    
    return  app_camera_poses_dict

def extract_poses_from_model_json(model_cameras_file_path):
    """
    The way we parse the json is from cameras.json at https://drive.google.com/drive/folders/1WEmUA2hh8NbIx2r590FDfHq-bedKIZdK 
    """
    # Open the json file 
    with open(model_cameras_file_path, 'r') as file:
        # Load the data from the file 
        model_camera_data = json.load(file)
    num_frames = len(model_camera_data)
    # number_pattern = ".*-(\d+)$"
    number_pattern = "(\d+)$"
    # import pdb
    # pdb.set_trace()
    model_camera_poses = []
    model_camera_poses_dict = {}

    for i in range(num_frames):
        frame = model_camera_data[i]
        frame_id = frame[MODEL_IMG_PATH]
        frame_id = re.findall(number_pattern, frame_id, re.IGNORECASE)
        frame_id = frame_id[0]
        
        position_matrix = frame[MODEL_POSITION_MATRIX]
        rotation_matrix = frame[MODEL_ROTATION_MATRIX]

        x_axis = rotation_matrix[0]
        y_axis = rotation_matrix[1]
        z_axis = rotation_matrix[2]

        pose_matrix = np.eye(4)
        pose_matrix[:3,0] = x_axis
        pose_matrix[:3,1] = y_axis
        pose_matrix[:3,2] = z_axis
        pose_matrix[:3,3] = position_matrix

        model_camera_poses.append(pose_matrix)
        model_camera_poses_dict[frame_id] = pose_matrix
    
    model_camera_poses = np.array(model_camera_poses)
    return model_camera_poses_dict

# Can be used to check the mapping with a given tolerance, currently (02/13/2024) this is not used 
def check_mapping(app_pose_array, model_pose_array, mapping, tolerance=25e-2):
    num_poses = app_pose_array.shape[0]
    allCloseToZero = True
    
    for idx in range(num_poses):
        app_pose = app_pose_array[idx].flatten()
        model_pose_est = np.matmul(app_pose, mapping)
        model_pose_orig = model_pose_array[idx].flatten()
        diff = model_pose_orig - model_pose_est

        allCloseToZero = np.allclose(diff, 0, atol=tolerance)
        if not allCloseToZero:
            
            return False 
        
    return True

# This function takes in a matrix in the camera coordinate system and puts it into the gaussian splatting coordinate system
def perform_mapping(app_pose_array, mapping):
    """
    Maps the app poses array to be in the 3D space of the model (this mapping is an estimate and it doesn't need to be precise)
    Arguments:
    - app_poses_array: a Nx4x4 matrix where N is the number of poses
    - mapping: a 16x1 matrix
    Return:
    - model_poses_array: a Nx4x4 matrix 
    """
    # Initialize return array
    num_poses = app_pose_array.shape[0]
    model_pose_est_array = []

    # Map each camera pose and add it to return array
    for idx in range(num_poses):
        app_pose = app_pose_array[idx].flatten()
        model_pose_est = np.matmul(app_pose, mapping)
        model_pose_est = model_pose_est.reshape(4, 4)
        model_pose_est_array.append(model_pose_est)

    return np.array(model_pose_est_array)

def create_camera_path_like_model(camera_poses_array, width=1902, height=1424, fx =1584.101, fy =1582.54, output_path="written_model_cameras.json"):
    """
    This function creates the camera path in a format that is the same as how the model saves the camera positions and rotations of the input camera poses
    """
    json_object = []
    count = 0
    for pose in camera_poses_array:
        camera_json_info = {}
        camera_json_info['id'] = count 
        camera_json_info['width'] = width
        camera_json_info['height'] = height
        camera_json_info['fx'] = fx
        camera_json_info['fy'] = fy
        camera_json_info['img_name'] = str(count)        
        camera_json_info['position'] = np.squeeze(pose[:3,3]).tolist()
        camera_json_info['rotation'] = pose[:3,:3].reshape(3, 3).tolist()

        # print(camera_json_info['position'])
        # print(camera_json_info['rotation'])

        json_object.append(camera_json_info)

        count+=1
        
    return json_object

def get_homography(source_coordinates, destination_coordinates):
    """ 
    Arguments:
    - Source coordinates: a Nx3 array
    - Desintation coordinates: a Nx3 array
    """
    # We must put our image coordinates into a matrix A
    A = np.zeros((3*source_coordinates.shape[0], 16))
    print(A.shape)
    NUM_VARS = 3
    # Now we must populate A with the proper values
    for i in range(A.shape[0]):
        coord_idx = i // NUM_VARS
        mult = 0
        if i % NUM_VARS == 0:
            A[i, 0] = source_coordinates[coord_idx, 0] # Col 0 corresponds to the x coordinate
            A[i, 1] = source_coordinates[coord_idx, 1] # Col 1 corresponds to the y coordinate
            A[i, 2] = source_coordinates[coord_idx, 2] # Col 2 corresponds to the z coordinate
            A[i, 3] = 1
            mult = destination_coordinates[coord_idx, 0] # Row 0 cooresponds to the u coordinate 
        elif i % 3 == 1:
            A[i, 4] = source_coordinates[coord_idx, 0] # Col 0 corresponds to the x coordinate 
            A[i, 5] = source_coordinates[coord_idx, 1] # Col 1 corresponds to the y coordinate
            A[i, 6] = source_coordinates[coord_idx, 2] # Col 2 corresponds to the z coordinate
            A[i, 7] = 1
            mult = destination_coordinates[coord_idx, 1] # Row 0 cooresponds to the u coordinate 
        else:
            A[i, 8] = source_coordinates[coord_idx, 0] # Col 0 corresponds to the x coordinate 
            A[i, 9] = source_coordinates[coord_idx, 1] # Col 1 corresponds to the y coordinate
            A[i, 10] = source_coordinates[coord_idx, 2] # Col 2 corresponds to the z coordinate
            A[i, 11] = 1
            mult = destination_coordinates[coord_idx, 2] # Row 0 cooresponds to the u coordinate 

        
        A[i, 12] = -mult*source_coordinates[coord_idx, 0]
        A[i, 13] = -mult*source_coordinates[coord_idx, 1]
        A[i, 14] = -mult*source_coordinates[coord_idx, 2]
        A[i, 15] = -mult

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(np.matmul(A.T, A))

    min_eigen_index = np.argmin(eigenvalues[eigenvalues != 0])
    # print(min_eigen_index)
    # print("eigenvalues:", eigenvalues)
    # print("eigenvectors:",eigenvectors)
    min_eigen_vector = eigenvectors[:,min_eigen_index]
    # print("min_eigen_vector")
    # print(min_eigen_vector.shape)
    # print("min_eigen_vector:", min_eigen_vector)
    projection_matrix = min_eigen_vector.reshape(4, 4)
    # print("projection_matrix:", projection_matrix)

    return projection_matrix

def create_homogenous_coordinates(positions_array):
    """
    Turns each position in positions_array into a homogenous coordinate
    Argument(s):
    - positions_array, Nx3 where N is the number of images
    Return 
    - Nx4 array where the 4th index is just 1 (representing a homgenous coordinate)
    """
    ones = np.ones((positions_array.shape[0], 1))
    homogenous_coords = np.concatenate([positions_array, ones], axis=1)
    return homogenous_coords

def apply_homography(points, homography):
    """
    Apply a homography transformation to a set of points.
    
    Args:
        points (ndarray): An nx3 numpy array representing the points.
        homography (ndarray): A 4x4 numpy array representing the homography matrix.
        
    Returns:
        ndarray: An nx3 numpy array containing the transformed points.
    """
    # Convert points to homogeneous coordinates
    points_homogeneous = create_homogenous_coordinates(points)
    
    # Apply homography transformation
    transformed_points_homogeneous = np.matmul(homography, points_homogeneous.T).T
    
    # Convert back to Cartesian coordinates
    transformed_points = transformed_points_homogeneous[:, :3] / transformed_points_homogeneous[:, 3:]
    
    return transformed_points

def refine_transformation_matrix(source_points, target_points, transformation_matrix):
    """ 
    source_points and target_points are Nx3 numpy arrays
    """
    # Convert NumPy arrays to Open3D point cloud objects
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_points)
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_points)

    # Perform point cloud registration
    transformation = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, 1000, transformation_matrix,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # Apply the transformation to source cloud to get it in the target point clouds frame
    source_cloud.transform(transformation.transformation)
    
    transformation_matrix_refined = transformation.transformation
    
    transformed_points = np.asarray(source_cloud.points)
    after_error = np.sum((transformed_points-target_points)**2)

    target_points_est_pre_refinement = apply_homography(source_points, transformation_matrix)
    before_error = np.sum((target_points_est_pre_refinement - target_points)**2)
    print("Before Error:", before_error)
    print("Error After: ", after_error)
    print("Transformation matrix:\n", transformation_matrix_refined)
    
    return transformation_matrix_refined, transformed_points, source_cloud, target_cloud

def findHomography(source_positions, target_positions):
    initial_homography = get_homography(source_positions, target_positions)
    initial_homography = refine_transformation_matrix(source_positions, target_positions, initial_homography)
    return initial_homography

def extract_img_coordinates_from_pose_dict(poses_dict):
    positions_dict = {}
    COORDINATE_COL = 3

    for key in poses_dict:
        camera_pose_matrix = poses_dict[key]
        coordinates = np.array([camera_pose_matrix[0][COORDINATE_COL], camera_pose_matrix[1][COORDINATE_COL], camera_pose_matrix[2][COORDINATE_COL]])
        positions_dict[key] = coordinates

    return positions_dict

def get_corresponding_arrays(app_camera_poses_dict, model_camera_poses_dict):
    # First init the lists we will return 
    app_camera_poses_array = []
    model_camera_poses_array = []

    for frame_id in app_camera_poses_dict:
        if frame_id in model_camera_poses_dict:
            app_camera_poses_array.append(app_camera_poses_dict[frame_id])
            model_camera_poses_array.append(model_camera_poses_dict[frame_id])
    
    app_camera_poses_array = np.array(app_camera_poses_array)
    model_camera_poses_array = np.array(model_camera_poses_array)

    return app_camera_poses_array, model_camera_poses_array

def get_corresponding_coordinates(app_poses_path, model_poses_path):
    app_poses_dict = extract_poses_from_app_json(app_poses_path)
    model_poses_dict = extract_poses_from_model_json(model_poses_path)

    app_positions_dict = extract_img_coordinates_from_pose_dict(app_poses_dict)
    model_positions_dict = extract_img_coordinates_from_pose_dict(model_poses_dict)

    app_positions_array, model_positions_array = get_corresponding_arrays(app_positions_dict, model_positions_dict)
    
    return app_positions_array, model_positions_array 

def draw_registration_point_cloud_bbox(source_array, source_bbox, target_array, target_bbox):
    source_bbox_temp = o3d.geometry.PointCloud()
    source_bbox_temp.points = o3d.utility.Vector3dVector(source_bbox)
    target_bbox_temp = o3d.geometry.PointCloud()
    target_bbox_temp.points = o3d.utility.Vector3dVector(target_bbox)
    source_temp = o3d.geometry.PointCloud()
    source_temp.points = o3d.utility.Vector3dVector(source_array)
    target_temp = o3d.geometry.PointCloud()
    target_temp.points = o3d.utility.Vector3dVector(target_array)
    source_temp.paint_uniform_color([186.0/255.0, 99.0/255.0, 0])
    source_bbox_temp.paint_uniform_color([191.0/255.0, 6/255.0, 141/255.0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    target_bbox_temp.paint_uniform_color([0, 0.03, 191.0/255.0])
    o3d.visualization.draw_geometries([source_temp, target_temp, source_bbox_temp, target_bbox_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])
    
def draw_registration_6_things(source_array, source_bbox, source_path, target_array, target_bbox, target_path):
    source_path_temp = o3d.geometry.PointCloud()
    source_path_temp.points = o3d.utility.Vector3dVector(source_path)
    target_path_temp = o3d.geometry.PointCloud()
    target_path_temp.points = o3d.utility.Vector3dVector(target_path)
    source_bbox_temp = o3d.geometry.PointCloud()
    source_bbox_temp.points = o3d.utility.Vector3dVector(source_bbox)
    target_bbox_temp = o3d.geometry.PointCloud()
    target_bbox_temp.points = o3d.utility.Vector3dVector(target_bbox)
    source_temp = o3d.geometry.PointCloud()
    source_temp.points = o3d.utility.Vector3dVector(source_array)
    target_temp = o3d.geometry.PointCloud()
    target_temp.points = o3d.utility.Vector3dVector(target_array)
    source_temp.paint_uniform_color([186.0/255.0, 99.0/255.0, 0])
    source_bbox_temp.paint_uniform_color([191.0/255.0, 6/255.0, 141/255.0])
    source_path_temp.paint_uniform_color([110.0/255.0, 28/255.0, 217/255.0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    target_bbox_temp.paint_uniform_color([0, 0.03, 191.0/255.0])
    target_path_temp.paint_uniform_color([21/255.0, 138/255.0, 108/255.0])
    o3d.visualization.draw_geometries([source_temp, target_temp, source_bbox_temp, target_bbox_temp, source_path_temp, target_path_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def get_bounding_box_points_as_numpy(bounding_box_file_path):
    # Open the .json file for reading
    with open(bounding_box_file_path, 'r') as file:
        # Load the data from the file
        bbox_json_data = json.load(file)

    # Cause there are 8 corners to a rectangular prism
    bbox_numpy = np.zeros((8, 3))
    idx = 0
    for point_name in bbox_json_data['positions']:
        point = bbox_json_data['positions'][point_name]
        x, y, z = point['x'], point['y'], point['z']
        
        bbox_numpy[idx] = np.array([x, y, z])
        idx += 1
        
    return bbox_numpy

def rotation_matrix_from_points(pointA, pointB):
    """
    Calculate the rotation matrix that aligns pointA with pointB.

    :param pointA: The original point.
    :param pointB: The target point.
    :return: The rotation matrix.
    """
    # Calculate the direction vector
    direction = pointB - pointA
    direction /= np.linalg.norm(direction)

    # Calculate the rotation vector (pitch and yaw only)
    rotation_vector = np.cross([0, 0, 1], direction)
    rotation_vector[2] = 0  # Zeroing out the z-component to eliminate roll rotation
    rotation_angle = np.arccos(np.dot([0, 0, 1], direction))

    # Create a rotation object from the rotation vector
    rotation = Rotation.from_rotvec(rotation_angle * rotation_vector)

    # Convert the rotation object to a rotation matrix
    rotation_matrix = rotation.as_matrix()

    return rotation_matrix

def rotation_matrix_from_pointsV3(pointA, pointB, normal_vec):
    """
    Calculate the rotation matrix that aligns pointA with pointB.

    :param pointA: The original point.
    :param pointB: The target point.
    :return: The rotation matrix.
    """
    zero_threshold = 1e-12
    # Calculate the direction vector
    direction = pointB - pointA
    direction /= np.linalg.norm(direction)

    normal_vec = np.array([0, 0, 1])

    # Calculate the rotation vector (pitch and yaw only)
    rotation_vector = np.cross(normal_vec, direction)
    # rotation_vector[2] = 0  # Zeroing out the z-component to eliminate roll rotation
    rotation_angle = np.arccos(np.dot(normal_vec, direction))

    # Create a rotation object from the rotation vector
    rotation = Rotation.from_rotvec(rotation_angle * rotation_vector)

    # Convert the rotation object to a rotation matrix
    rotation_matrix = rotation.as_matrix()
    mask = np.abs(rotation_matrix) < zero_threshold
    rotation_matrix[mask] = 0.
    print("REGULAR ROTATION MATRIX: ", rotation_matrix)

    return rotation_matrix

def rotation_matrix_from_pointsV4(pointA, pointB, normal_vec):
    """
    Calculate the rotation matrix that aligns pointA with pointB.

    :param pointA: The original point.
    :param pointB: The target point.
    :return: The rotation matrix.
    """

    # Calculate the direction vector
    direction = pointB - pointA
    vec2 = direction / np.linalg.norm(direction)
    up = normal_vec # np.array([0, 0, 1])
    vec0 = np.cross(up, direction)
    vec1 = np.cross(vec2, vec0)
    rotation_matrix = np.stack([vec0, vec1, vec2], axis=1)
    return rotation_matrix

# def rotation_matrix_from_points(pointA, pointB):
#     """
#     Calculate the rotation matrix that aligns pointA with pointB.

#     :param pointA: The original point.
#     :param pointB: The target point.
#     :return: The rotation matrix.
#     """
#     # Calculate the direction vector
#     direction = pointB - pointA
#     direction /= np.linalg.norm(direction)

#     # Calculate the rotation vector
#     rotation_vector = np.cross([0, 0, 1], direction)
#     rotation_angle = np.arccos(np.dot([0, 0, 1], direction))

#     # Create a rotation object from the rotation vector
#     rotation = Rotation.from_rotvec(rotation_angle * rotation_vector)

#     # Convert the rotation object to a rotation matrix
#     rotation_matrix = rotation.as_matrix()

#     return rotation_matrix

# def rotation_matrix_from_points(pointA, pointB):
#     """
#     Calculate the rotation matrix that aligns a vector from pointA to pointB.

#     :param pointA: The original point.
#     :param pointB: The target point.
#     :return: The rotation matrix.
#     """
#     # Calculate the direction vector
#     direction = pointB - pointA
#     direction /= np.linalg.norm(direction)

#     # Calculate rotation matrix components
#     # First, calculate the rotation around the z-axis (yaw)
#     yaw = np.arctan2(direction[1], direction[0])
#     yaw_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
#                            [np.sin(yaw), np.cos(yaw), 0],
#                            [0, 0, 1]])

#     # Then, calculate the rotation around the y-axis (pitch)
#     pitch = np.arcsin(direction[2])
#     pitch_matrix = np.array([[np.cos(pitch), 0, np.sin(pitch)],
#                              [0, 1, 0],
#                              [-np.sin(pitch), 0, np.cos(pitch)]])

#     # Combine both rotations
#     rotation_matrix = np.dot(yaw_matrix, pitch_matrix)

#     return rotation_matrix

# def rotation_matrix_from_points(pointA, pointB):
#     """
#     Calculate the rotation matrix that aligns pointA with pointB in terms of pitch and yaw.

#     :param pointA: The original point.
#     :param pointB: The target point.
#     :return: The rotation matrix.
#     """
#     # Normalize the points
#     pointA_norm = pointA / np.linalg.norm(pointA)
#     pointB_norm = pointB / np.linalg.norm(pointB)

#     # Align the z-axis of pointA with pointB
#     axis_z = np.cross(pointA_norm, pointB_norm)
#     angle_z = np.arcsin(np.linalg.norm(axis_z))

#     # Create a rotation around the z-axis
#     rotation_z = Rotation.from_rotvec(angle_z * (axis_z / np.linalg.norm(axis_z)))

#     # Rotate pointA to the same orientation as pointB around the z-axis
#     pointA_aligned = rotation_z.apply(pointA)

#     # Calculate the yaw angle around the z-axis
#     angle_yaw = np.arctan2(pointB[1], pointB[0]) - np.arctan2(pointA_aligned[1], pointA_aligned[0])

#     # Create a rotation around the z-axis for yaw
#     rotation_yaw = Rotation.from_rotvec(angle_yaw * np.array([0, 0, 1]))

#     # Combine the rotations
#     rotation = rotation_yaw * rotation_z

#     # Convert the rotation object to a rotation matrix
#     rotation_matrix = rotation.as_matrix()
    
#     return rotation_matrix

def extract_coordinates(transformation_matrix):
    if len(transformation_matrix.shape) == 2:
        coordinate_col = 3
        coordinates = (transformation_matrix[0][coordinate_col], transformation_matrix[1][coordinate_col], transformation_matrix[2][coordinate_col])
    else:
        coordinates = np.zeros((transformation_matrix.shape[0], 3))
        coordinates = transformation_matrix[:,:3,3]
    return coordinates

def get_camera_matrix_pointing_center(camera_position, bbox, normal_vec):
    center = np.array([bbox[0], bbox[1], bbox[2]])
    # rotation_matrix = rotation_matrix_from_points(camera_position, center) 
    rotation_matrix = rotation_matrix_from_pointsV4(camera_position, center, normal_vec)# rotation_matrix_from_pointsV3(camera_position, center, normal_vec) 
    # rotation_matrix = get_rotation_angles(center, camera_position)
    
    # camera_pose_matrix = create_camera_pose_matrix(camera_position[0], camera_position[1], camera_position[2], (math.pi-rotation_angles[0]), (math.pi-rotation_angles[1]))
    camera_pose_matrix = np.eye(4)
    camera_pose_matrix[:3,:3] = rotation_matrix
    camera_pose_matrix[0, 3] = camera_position[0]
    camera_pose_matrix[1, 3] = camera_position[1]
    camera_pose_matrix[2, 3] = camera_position[2]
    
    return camera_pose_matrix

def get_camera_poses_from_positions_and_center(camera_positions, bbox, normal_vec):
    camera_poses = np.ones([camera_positions.shape[0], 4, 4])
    print("BBOX: ", bbox)
    # This part can be optimized by making more use of np.arrays
    for i in range(camera_positions.shape[0]):
        camera_position = np.array([camera_positions[i][0], camera_positions[i][1], camera_positions[i][2]])
        camera_pose_matrix = get_camera_matrix_pointing_center(camera_position, bbox, normal_vec)
        camera_poses[i] = camera_pose_matrix
    return camera_poses

# def get_camera_poses_from_positions_and_center_nerfstudio(camera_positions, bbox, normal_vec):
#     camera_poses = np.ones([camera_positions.shape[0], 4, 4])
#     center = np.array([bbox[0], bbox[1], bbox[2]])
#     # print("BBOX: ", bbox)
#     # This part can be optimized by making more use of np.arrays
#     for i in range(camera_positions.shape[0]):
#         camera_position = np.array([camera_positions[i][0], camera_positions[i][1], camera_positions[i][2]])
#         lookat = torch.tensor((center - camera_position), dtype=torch.float32)
#         up = torch.tensor(np.array([0, 0, 1]), dtype=torch.float32)
#         pos = torch.tensor(camera_position, dtype=torch.float32)
#         camera_pose_matrix = viewmatrix(lookat, up, pos).numpy()
#         camera_pose_matrix = np.concatenate((camera_pose_matrix, np.array([[0, 0, 0, 1]])), axis=0)
        
#         camera_poses[i] = camera_pose_matrix
#     return camera_poses

def get_bounding_box_dict_points_as_numpy(bbox_json_data):

    points = []
    for point in bbox_json_data['positions']:
        if point != 'center':
            points.append(np.array([bbox_json_data['positions'][point]['x'], bbox_json_data['positions'][point]['y'], bbox_json_data['positions'][point]['z']] ))
    
    return np.array(points)

def get_bounding_box_points_as_dict(bounding_box_file_path):
    # Open the .json file for reading
    with open(bounding_box_file_path, 'r') as file:
        # Load the data from the file
        bbox_json_data = json.load(file)

    # Cause there are 8 corners to a rectangular prism
    bbox_dict_json = {}

    bbox_dict_arrays = {}
    idx = 0
    for point_name in bbox_json_data['positions']:
        point = bbox_json_data['positions'][point_name]
        x, y, z = point['x'], point['y'], point['z']
        
        bbox_dict_json[point_name] = {}
        bbox_dict_json[point_name]['x'] = x
        bbox_dict_json[point_name]['y'] = y
        bbox_dict_json[point_name]['z'] = z

        bbox_dict_arrays[point_name] = np.array([x, y, z])
        idx += 1
    
    bbox_dict_json['center'] = copy.deepcopy(bbox_json_data['center'])
    center_point = np.array([bbox_json_data['center']['x'], bbox_json_data['center']['y'], bbox_json_data['center']['z']])
    bbox_dict_arrays['center'] = center_point

    return bbox_dict_json, bbox_dict_arrays

def get_bounding_box_replica_other_frame(bbox_original_frame_path, homography):
    bbox_app_json, bbox_app_array = get_bounding_box_points_as_dict(bbox_original_frame_path)

    bbox_model_frame = {}
    bbox_model_frame['positions'] = {}
    for point_name in bbox_app_array:
        point = bbox_app_array[point_name]
        point_model_frame = apply_homography(point.reshape(1, -1), homography)
        point_model_frame = np.squeeze(point_model_frame).tolist()
        bbox_model_frame['positions'][point_name] = {}
        bbox_model_frame['positions'][point_name]['x'] = point_model_frame[0]  
        bbox_model_frame['positions'][point_name]['y'] = point_model_frame[1] 
        bbox_model_frame['positions'][point_name]['z'] = point_model_frame[2] 

    point = bbox_app_array['center']
    point_model_frame = apply_homography(point.reshape(1, -1), homography)
    point_model_frame = np.squeeze(point_model_frame).tolist()
    bbox_model_frame['center'] = {}
    bbox_model_frame['center']['x'] = point_model_frame[0]
    bbox_model_frame['center']['y'] = point_model_frame[1]
    bbox_model_frame['center']['z'] = point_model_frame[2]
    return bbox_model_frame

# def normalize(x: torch.Tensor) -> Float[Tensor, "*batch"]:
#     """Returns a normalized vector."""
#     return x / torch.linalg.norm(x)

# def viewmatrix(lookat: torch.Tensor, up: torch.Tensor, pos: torch.Tensor) -> Float[Tensor, "*batch"]:
#     """Returns a camera transformation matrix.

#     Args:
#         lookat: The direction the camera is looking.
#         up: The upward direction of the camera.
#         pos: The position of the camera.

#     Returns:
#         A camera transformation matrix.
#     """
#     vec2 = normalize(lookat)
#     vec1_avg = normalize(up)
#     vec0 = normalize(torch.cross(vec1_avg, vec2))
#     vec1 = normalize(torch.cross(vec2, vec0))
#     m = torch.stack([vec0, vec1, vec2, pos], 1)
#     return m

# _EPS = np.finfo(float).eps * 4.0

# def quaternion_from_matrix(matrix: NDArray, isprecise: bool = False) -> np.ndarray:
#     """Return quaternion from rotation matrix.

#     Args:
#         matrix: rotation matrix to obtain quaternion
#         isprecise: if True, input matrix is assumed to be precise rotation matrix and a faster algorithm is used.
#     """
#     M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
#     if isprecise:
#         q = np.empty((4,))
#         t = np.trace(M)
#         if t > M[3, 3]:
#             q[0] = t
#             q[3] = M[1, 0] - M[0, 1]
#             q[2] = M[0, 2] - M[2, 0]
#             q[1] = M[2, 1] - M[1, 2]
#         else:
#             i, j, k = 1, 2, 3
#             if M[1, 1] > M[0, 0]:
#                 i, j, k = 2, 3, 1
#             if M[2, 2] > M[i, i]:
#                 i, j, k = 3, 1, 2
#             t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
#             q[i] = t
#             q[j] = M[i, j] + M[j, i]
#             q[k] = M[k, i] + M[i, k]
#             q[3] = M[k, j] - M[j, k]
#         q *= 0.5 / math.sqrt(t * M[3, 3])
#     else:
#         m00 = M[0, 0]
#         m01 = M[0, 1]
#         m02 = M[0, 2]
#         m10 = M[1, 0]
#         m11 = M[1, 1]
#         m12 = M[1, 2]
#         m20 = M[2, 0]
#         m21 = M[2, 1]
#         m22 = M[2, 2]
#         # symmetric matrix K
#         K = [
#             [m00 - m11 - m22, 0.0, 0.0, 0.0],
#             [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
#             [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
#             [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
#         ]
#         K = np.array(K)
#         K /= 3.0
#         # quaternion is eigenvector of K that corresponds to largest eigenvalue
#         w, V = np.linalg.eigh(K)
#         q = V[np.array([3, 0, 1, 2]), np.argmax(w)]
#     if q[0] < 0.0:
#         np.negative(q, q)
#     return q

# def unit_vector(data: NDArray, axis: Optional[int] = None) -> np.ndarray:
#     """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

#     Args:
#         axis: the axis along which to normalize into unit vector
#         out: where to write out the data to. If None, returns a new np ndarray
#     """
#     data = np.array(data, dtype=np.float64, copy=True)
#     if data.ndim == 1:
#         data /= math.sqrt(np.dot(data, data))
#         return data
#     length = np.atleast_1d(np.sum(data * data, axis))
#     np.sqrt(length, length)
#     if axis is not None:
#         length = np.expand_dims(length, axis)
#     data /= length
#     return data

# def quaternion_slerp(
#     quat0: NDArray, quat1: NDArray, fraction: float, spin: int = 0, shortestpath: bool = True
# ) -> np.ndarray:
#     """Return spherical linear interpolation between two quaternions.
#     Args:
#         quat0: first quaternion
#         quat1: second quaternion
#         fraction: how much to interpolate between quat0 vs quat1 (if 0, closer to quat0; if 1, closer to quat1)
#         spin: how much of an additional spin to place on the interpolation
#         shortestpath: whether to return the short or long path to rotation
#     """
#     q0 = unit_vector(quat0[:4])
#     q1 = unit_vector(quat1[:4])
#     if q0 is None or q1 is None:
#         raise ValueError("Input quaternions invalid.")
#     if fraction == 0.0:
#         return q0
#     if fraction == 1.0:
#         return q1
#     d = np.dot(q0, q1)
#     if abs(abs(d) - 1.0) < _EPS:
#         return q0
#     if shortestpath and d < 0.0:
#         # invert rotation
#         d = -d
#         np.negative(q1, q1)
#     angle = math.acos(d) + spin * math.pi
#     if abs(angle) < _EPS:
#         return q0
#     isin = 1.0 / math.sin(angle)
#     q0 *= math.sin((1.0 - fraction) * angle) * isin
#     q1 *= math.sin(fraction * angle) * isin
#     q0 += q1
#     return q0

# def quaternion_matrix(quaternion: NDArray) -> np.ndarray:
#     """Return homogeneous rotation matrix from quaternion.

#     Args:
#         quaternion: value to convert to matrix
#     """
#     q = np.array(quaternion, dtype=np.float64, copy=True)
#     n = np.dot(q, q)
#     if n < _EPS:
#         return np.identity(4)
#     q *= math.sqrt(2.0 / n)
#     q = np.outer(q, q)
#     return np.array(
#         [
#             [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
#             [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
#             [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
#             [0.0, 0.0, 0.0, 1.0],
#         ]
#     )

# def get_ordered_poses_and_k(
#     poses: Float[Tensor, "num_poses 3 4"],
#     Ks: Float[Tensor, "num_poses 3 3"],
# ) -> Tuple[Float[Tensor, "num_poses 3 4"], Float[Tensor, "num_poses 3 3"]]:
#     """
#     Returns ordered poses and intrinsics by euclidian distance between poses.

#     Args:
#         poses: list of camera poses
#         Ks: list of camera intrinsics

#     Returns:
#         tuple of ordered poses and intrinsics

#     """

#     poses_num = len(poses)

#     ordered_poses = torch.unsqueeze(poses[0], 0)
#     ordered_ks = torch.unsqueeze(Ks[0], 0)

#     # remove the first pose from poses
#     poses = poses[1:]
#     Ks = Ks[1:]

#     for _ in range(poses_num - 1):
#         distances = torch.norm(ordered_poses[-1][:, 3] - poses[:, :, 3], dim=1)
#         idx = torch.argmin(distances)
#         ordered_poses = torch.cat((ordered_poses, torch.unsqueeze(poses[idx], 0)), dim=0)
#         ordered_ks = torch.cat((ordered_ks, torch.unsqueeze(Ks[idx], 0)), dim=0)
#         poses = torch.cat((poses[0:idx], poses[idx + 1 :]), dim=0)
#         Ks = torch.cat((Ks[0:idx], Ks[idx + 1 :]), dim=0)

#     return ordered_poses, ordered_ks

# def get_ordered_poses(
#     poses: Float[Tensor, "num_poses 3 4"],
# ) -> Tuple[Float[Tensor, "num_poses 3 4"], Float[Tensor, "num_poses 3 3"]]:
#     """
#     Returns ordered poses and intrinsics by euclidian distance between poses.

#     Args:
#         poses: list of camera poses
#         Ks: list of camera intrinsics

#     Returns:
#         tuple of ordered poses and intrinsics

#     """

#     poses_num = len(poses)

#     ordered_poses = torch.unsqueeze(poses[0], 0)

#     # remove the first pose from poses
#     poses = poses[1:]

#     for _ in range(poses_num - 1):
#         distances = torch.norm(ordered_poses[-1][:, 3] - poses[:, :, 3], dim=1)
#         idx = torch.argmin(distances)
#         ordered_poses = torch.cat((ordered_poses, torch.unsqueeze(poses[idx], 0)), dim=0)
#         poses = torch.cat((poses[0:idx], poses[idx + 1 :]), dim=0)

#     return ordered_poses


# def get_interpolated_k(
#     k_a: Float[Tensor, "3 3"], k_b: Float[Tensor, "3 3"], steps: int = 10
# ) -> List[Float[Tensor, "3 4"]]:
#     """
#     Returns interpolated path between two camera poses with specified number of steps.

#     Args:
#         k_a: camera matrix 1
#         k_b: camera matrix 2
#         steps: number of steps the interpolated pose path should contain

#     Returns:
#         List of interpolated camera poses
#     """
#     Ks: List[Float[Tensor, "3 3"]] = []
#     ts = np.linspace(0, 1, steps)
#     for t in ts:
#         new_k = k_a * (1.0 - t) + k_b * t
#         Ks.append(new_k)
#     return Ks

# def get_interpolated_poses(pose_a: NDArray, pose_b: NDArray, steps: int = 10) -> List[float]:
#     """Return interpolation of poses with specified number of steps.
#     Args:
#         pose_a: first pose
#         pose_b: second pose
#         steps: number of steps the interpolated pose path should contain
#     """

#     quat_a = quaternion_from_matrix(pose_a[:3, :3])
#     quat_b = quaternion_from_matrix(pose_b[:3, :3])

#     ts = np.linspace(0, 1, steps)
#     quats = [quaternion_slerp(quat_a, quat_b, t) for t in ts]
#     trans = [(1 - t) * pose_a[:3, 3] + t * pose_b[:3, 3] for t in ts]

#     poses_ab = []
#     for quat, tran in zip(quats, trans):
#         pose = np.identity(4)
#         pose[:3, :3] = quaternion_matrix(quat)[:3, :3]
#         pose[:3, 3] = tran
#         poses_ab.append(pose[:3])
#     return poses_ab

# def get_interpolated_poses_many(
#     poses: Float[Tensor, "num_poses 3 4"],
#     Ks: Float[Tensor, "num_poses 3 3"],
#     steps_per_transition: int = 10,
#     order_poses: bool = False,
# ) -> Tuple[Float[Tensor, "num_poses 3 4"], Float[Tensor, "num_poses 3 3"]]:
#     """Return interpolated poses for many camera poses.

#     Args:
#         poses: list of camera poses
#         Ks: list of camera intrinsics
#         steps_per_transition: number of steps per transition
#         order_poses: whether to order poses by euclidian distance

#     Returns:
#         tuple of new poses and intrinsics
#     """
#     traj = []
#     k_interp = []

#     if order_poses:
#         poses, Ks = get_ordered_poses_and_k(poses, Ks)

#     for idx in range(poses.shape[0] - 1):
#         pose_a = poses[idx].cpu().numpy()
#         pose_b = poses[idx + 1].cpu().numpy()
#         poses_ab = get_interpolated_poses(pose_a, pose_b, steps=steps_per_transition)
#         traj += poses_ab
#         k_interp += get_interpolated_k(Ks[idx], Ks[idx + 1], steps=steps_per_transition)

#     traj = np.stack(traj, axis=0)
#     k_interp = torch.stack(k_interp, dim=0)

#     return torch.tensor(traj, dtype=torch.float32), torch.tensor(k_interp, dtype=torch.float32)


# def get_interpolated_poses_many_josh(
#     poses: Float[Tensor, "num_poses 3 4"],
#     steps_per_transition: int = 10,
#     order_poses: bool = False,
# ) -> Tuple[Float[Tensor, "num_poses 3 4"], Float[Tensor, "num_poses 3 3"]]:
#     """Return interpolated poses for many camera poses.

#     Args:
#         poses: list of camera poses
#         Ks: list of camera intrinsics
#         steps_per_transition: number of steps per transition
#         order_poses: whether to order poses by euclidian distance

#     Returns:
#         tuple of new poses and intrinsics
#     """
#     traj = []

#     if order_poses:
#         poses, Ks = get_ordered_poses(poses)

#     for idx in range(poses.shape[0] - 1):
#         pose_a = poses[idx].cpu().numpy()
#         pose_b = poses[idx + 1].cpu().numpy()
#         poses_ab = get_interpolated_poses(pose_a, pose_b, steps=steps_per_transition)
#         traj += poses_ab

#     traj = np.stack(traj, axis=0)

#     return torch.tensor(traj, dtype=torch.float32)

# PRODUCTION CAMERA PATH FUNCTION
def create_camera_path_json(bounding_box_file_path, app_camera_poses_path, model_camera_poses_path, num_points=100, output_path="written_camera_path.json"):
    # Get corresponding app and model positions
    app_positions_array, model_positions_array = get_corresponding_coordinates(app_camera_poses_path, model_camera_poses_path)
    homography, _, _, _ = findHomography(app_positions_array, model_positions_array)
    # Get bounding box and camerea poses in app frame
    bbox = get_bounding_box(bounding_box_file_path)
    camera_poses_app = get_camera_poses(bbox, create_snaking_line, num_points)
    camera_poses_model = np.matmul(homography, camera_poses_app)
    camera_poses_model = camera_poses_model / camera_poses_model[:,3:,3:]
    
    bbox_model_frame = get_bounding_box_replica_other_frame(bounding_box_file_path, homography)
    bbox_model_frame_numpy = get_bounding_box_from_dict(bbox_model_frame)
    normal_vec = -1*np.array([bbox_model_frame['positions']["top_right_back"]['x']-bbox_model_frame['positions']["bot_right_back"]['x'] , bbox_model_frame['positions']["top_right_back"]['y']-bbox_model_frame['positions']["bot_right_back"]['y'] , bbox_model_frame['positions']["top_right_back"]['z']-bbox_model_frame['positions']["bot_right_back"]['z'] ])
    normal_vec /= np.linalg.norm(normal_vec)
    camera_positions_model = extract_coordinates(camera_poses_model)
    # The below variable has the camera poses for the camera path in the model's frame
    camera_poses_model = get_camera_poses_from_positions_and_center(camera_positions_model, bbox_model_frame_numpy, normal_vec)

    json_info = create_camera_path_like_model(camera_poses_model)

    write_json(json_info, output_path=output_path)

# # PRODUCTION 2 CAMERA PATH FUNCTION
# def create_camera_path_json_nerfstudio(bounding_box_file_path, app_camera_poses_path, model_camera_poses_path, num_points=100, output_path="written_camera_path.json"):
#     # Get corresponding app and model positions
#     app_positions_array, model_positions_array = get_corresponding_coordinates(app_camera_poses_path, model_camera_poses_path)
#     homography, _, _, _ = findHomography(app_positions_array, model_positions_array)
#     # Get bounding box and camerea poses in app frame
#     bbox = get_bounding_box(bounding_box_file_path)
#     camera_poses_app = get_camera_poses(bbox, create_snaking_line, num_points)
#     camera_poses_model = np.matmul(homography, camera_poses_app)
#     camera_poses_model = camera_poses_model / camera_poses_model[:,3:,3:]
    
#     bbox_model_frame = get_bounding_box_replica_other_frame(bounding_box_file_path, homography)
#     bbox_model_frame_numpy = get_bounding_box_from_dict(bbox_model_frame)
#     normal_vec = -1*np.array([bbox_model_frame['positions']["top_right_back"]['x']-bbox_model_frame['positions']["bot_right_back"]['x'] , bbox_model_frame['positions']["top_right_back"]['y']-bbox_model_frame['positions']["bot_right_back"]['y'] , bbox_model_frame['positions']["top_right_back"]['z']-bbox_model_frame['positions']["bot_right_back"]['z'] ])
#     normal_vec /= np.linalg.norm(normal_vec)
#     camera_positions_model = get_bounding_box_dict_points_as_numpy(bbox_model_frame)
    
#     # The below variable has the camera poses for the camera path in the model's frame
#     camera_poses_model = get_camera_poses_from_positions_and_center_nerfstudio(camera_positions_model, bbox_model_frame_numpy, normal_vec)
#     camera_poses_model = camera_poses_model[:,:3,:]

#     camera_poses_model_torch = torch.tensor(camera_poses_model, dtype=torch.float32)

#     interpolated_poses = get_interpolated_poses_many_josh(camera_poses_model_torch)

#     interpolated_poses = interpolated_poses.numpy()

#     bottom_row = np.zeros((interpolated_poses.shape[0], 1, 4))
#     bottom_row[:,0,3] = 1

#     interpolated_poses = np.concatenate((interpolated_poses, bottom_row), axis=1)
    
#     camera_poses_model = interpolated_poses

#     json_info = create_camera_path_like_model(camera_poses_model)

#     write_json(json_info, output_path=output_path)

# Example: python3 get_bounding_box_model_frame.py boundingbox.json transforms.json cameras.json output_path.json 100
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script.')

    # Define command-line arguments
    parser.add_argument('app_bounding_box', type=str, help='The path to the bounding box json file.')
    parser.add_argument('app_camera_poses', type=str, help='The path to the app\'s camera poses json file.')
    parser.add_argument('model_camera_poses', type=str, help='The path to the model\'s camera path json file.')
    parser.add_argument('camera_path_output', type=str, help='The output path of the camera path json file.')
    parser.add_argument('num_camera_points', type=int, help='The number of camera path points.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the arguments
    bounding_box_file_path = args.app_bounding_box
    app_camera_poses_path = args.app_camera_poses
    model_camera_poses_path = args.model_camera_poses
    output_path = args.camera_path_output
    num_interpolated_points = args.num_camera_points
    
    create_camera_path_json(bounding_box_file_path, app_camera_poses_path, model_camera_poses_path, num_points=num_interpolated_points, output_path=output_path)
    
