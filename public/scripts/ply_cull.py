import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d
import json
import argparse
from itertools import product as iterprod
import pdb # for debugging

# for testing only
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
Using Open3D crop functionality to cull points.

Reference:
    https://www.open3d.org/docs/release/python_api/open3d.geometry.OrientedBoundingBox.html#open3d.geometry.OrientedBoundingBox.create_from_points
    https://stackoverflow.com/questions/71051649/select-remaining-points-after-cropping-a-point-cloud

Notes:
    - o3d.geometry.OrientedBoundingBox.create_from_points() may not be the most accurate solution.
'''

def cullPly(bbox, filename = "input.ply", saveTo = "output.ply"):
    '''
    Culls all points in a ply file outside of bounds and saves the new version. 

    bbox: open3d.geometry.OrientedBoundingBox. comes from readBounds()
    filename: String of filename of ply file to read.
    saveTo: String of filename ply file to write to.

    '''
    with open(filename, 'rb') as f:
        # OPEN FILE WITH OPEN3d
        ply_file_path = filename
        point_cloud = o3d.io.read_point_cloud(ply_file_path, format="ply")      

        # get index of all points inside bounds
        inside_indices = bbox.get_point_indices_within_bounding_box(point_cloud.points)

        # OPEN FILE WITH PLYFILE
        print("reading with plyfile")
        plydata = PlyData.read(f)
        print("finished reading, copying data")
        newData = PlyData(plydata.elements, text=False, byte_order = plydata.byte_order) # the file we will be working with
        # note that the webviewer will only accept the data as binary, not ascii. hence text=False
        
        numPoints = len(newData.elements[0].data)

        # cull outside_indices from data
        mask = np.zeros(numPoints, dtype=bool)
        mask[inside_indices] = True
        newData.elements[0].data = newData.elements[0].data[mask]

        # write culled ply file
        print(f"From {numPoints} points, culled {numPoints - len(newData.elements[0].data)} points. {len(newData.elements[0].data)} points remaining.")
        print(f"writing to {saveTo}")
        newData.write(saveTo)

def reducePoints(maxPoints, filename = "input.ply", saveTo = "output.ply"):
    '''
    Takes a ply file and reduces the number of points. Use for testing purposes only.
    '''
    with open(filename, 'rb') as f:
        # OPEN FILE WITH PLYFILE
        print("reading with plyfile")
        plydata = PlyData.read(f)
        print("finished reading, copying data")
        pdb.set_trace()
        newData = PlyData(plydata.elements, text=False, byte_order = plydata.byte_order) # the file we will be working with
        # note that the webviewer will only take the data as binary, not ascii. hence text=False
        
        # remove points
        newData.elements[0].data = newData.elements[0].data[:maxPoints]

        # write data
        print(f"writing to {saveTo}")
        newData.write(saveTo)

def readBounds(filepath = "boundingbox.json", debug = False):
    '''
    Reads in the bounding box json and outputs a open3d.geometry.OrientedBoundingBox.
    '''    
    positionNameParts = [["top","bot"],["left","right"],["back","front"]]

    # get list of 8 names like bot_left_back, etc.
    positionNames = ["_".join(r) for r in iterprod(positionNameParts[0],positionNameParts[1],positionNameParts[2])]
    
    with open(filepath) as f:
        data = json.load(f)

    # load in points
    points = [data[position] for position in positionNames]
    points = np.array(points)

    # make open3d.geometry.OrientedBoundingBox
    vec3d = o3d.utility.Vector3dVector(points)
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(vec3d)

    return bbox

# command line parser
# Example: python ply_cull.py test/original_point_cloud.ply -b test/model_boundingbox.json -o test/myout.ply
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("--bounding_box", "-b", help="Path to the bounding box file")
    parser.add_argument("--output", "-o", help="Path to the output file")

    args = parser.parse_args()

    bbox = readBounds(args.bounding_box, debug=False)
    cullPly(bbox, filename = args.input_file, saveTo=args.output)

if __name__ == "__main__":
    main()