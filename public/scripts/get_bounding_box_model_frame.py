import json
import re
import copy
import numpy as np
import open3d as o3d
import argparse

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

# Code adapted from Chat-GPT
def create_snaking_line(bbox, num_points, num_turns=3, radius_scale=2):
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
    # Translation matrix
    translation_matrix = np.array([[1, 0, 0, x],
                                   [0, 1, 0, y],
                                   [0, 0, 1, z],
                                   [0, 0, 0, 1]])

    # Rotation matrices
    rotation_y = np.array([[np.cos(pitch), 0, np.sin(pitch), 0],
                           [0, 1, 0, 0],
                           [-np.sin(pitch), 0, np.cos(pitch), 0],
                           [0, 0, 0, 1]])

    rotation_z = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                           [np.sin(yaw), np.cos(yaw), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

    # Combine translation and rotation
    pose_matrix = np.dot(np.dot(translation_matrix, rotation_z), rotation_y)

    return pose_matrix

# Function takes in a camera position and a bounding box and outputs the camera 6DoF coordinates in a 4x4 matrix for the camera to look at the closest point to the ellipsoid 
# that is defined by the bounding box
def get_camera_matrix(camera_position, bbox):
    closest_point = find_closest_point_on_ellipsoid(camera_position, bbox)
    rotation_angles = get_rotation_angles(closest_point, camera_position)
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
        camera_json_info['img_name'] = f"{count}-{count}"       
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

def get_rid_of_shear(homography):
    new_homography = np.copy(homography)
    new_homography[3,:3] = np.array([0, 0, 0])
    return new_homography

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
    
    # transformed_points = np.asarray(source_cloud.points)
    transformed_points = apply_homography(source_points, transformation_matrix_refined)
    after_error = np.sum((transformed_points-target_points)**2)

    target_points_est_pre_refinement = apply_homography(source_points, transformation_matrix)
    before_error = np.sum((target_points_est_pre_refinement - target_points)**2)
    print("Before Error:", before_error)
    print("Error After: ", after_error)
    print("Transformation matrix:\n", transformation_matrix_refined)

    print("Getting rid of shear:")
    transformation_matrix_refined = get_rid_of_shear(transformation_matrix_refined)
    target_points_est_refinement_shear = apply_homography(source_points, transformation_matrix_refined)
    after_error = np.sum((target_points_est_refinement_shear - target_points)**2)
    print("Error After w/o shear: ", after_error)
    print("Transformation matrix w/o shear:\n", transformation_matrix_refined)
    
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

def extract_coordinates(transformation_matrix):
    if len(transformation_matrix.shape) == 2:
        coordinate_col = 3
        coordinates = (transformation_matrix[0][coordinate_col], transformation_matrix[1][coordinate_col], transformation_matrix[2][coordinate_col])
    else:
        coordinates = np.zeros((transformation_matrix.shape[0], 3))
        coordinates = transformation_matrix[:,:3,3]
    return coordinates

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

def create_models_bbox_json(bounding_box_file_path, app_camera_poses_path, model_camera_poses_path, output_path="model_bbox.json"):
    # Get corresponding app and model positions
    app_positions_array, model_positions_array = get_corresponding_coordinates(app_camera_poses_path, model_camera_poses_path)
    homography, _, _, _ = findHomography(app_positions_array, model_positions_array)
    # Get bounding box and camerea poses in app frame
    bbox_app_json, bbox_app_array = get_bounding_box_points_as_dict(bounding_box_file_path)
    bbox_model_frame = {}
    for point_name in bbox_app_array:
        point = bbox_app_array[point_name]
        point_model_frame = apply_homography(point.reshape(1, -1), homography)
        bbox_model_frame[point_name] = np.squeeze(point_model_frame).tolist()
    
    point = bbox_app_array['center']
    point_model_frame = apply_homography(point.reshape(1, -1), homography)
    bbox_model_frame['center'] = np.squeeze(point_model_frame).tolist()
    write_json(bbox_model_frame, output_path=output_path)

# Example: python3 get_bounding_box_model_frame.py boundingbox.json transforms.json cameras.json output_path.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script.')

    # Define command-line arguments
    parser.add_argument('app_bounding_box', type=str, help='The path to the bounding box json file.')
    parser.add_argument('app_camera_poses', type=str, help='The path to the app\'s camera poses json file.')
    parser.add_argument('model_camera_poses', type=str, help='The path to the model\'s camera path json file.')
    parser.add_argument('bounding_box_output_path', type=str, help='The output path of the camera path json file.')
    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the arguments
    bounding_box_file_path = args.app_bounding_box
    app_camera_poses_path = args.app_camera_poses
    model_camera_poses_path = args.model_camera_poses
    output_path = args.bounding_box_output_path
    
    create_models_bbox_json(bounding_box_file_path, app_camera_poses_path, model_camera_poses_path, output_path=output_path)
    
