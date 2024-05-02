# Get Camera Poses & Bounding Box Python File Documentation
The below documentation will mainly be for the get camera poses python file because the bounding box python file is pretty much a subset of the camera poses one. 

## Input (Get Camera Poses)
This file takes in the bounding box file, the camera poses in the app's frame, the camera poses in the model's frame, the number of points to draw around the bounding box, the number, and the file name of the output.

## How it works (Get Camera Poses)
We save the bounding box file and both the camera pose files as a dictinary in python and extract their positions. We save these positions into two arrays where index i in both arrays corresponds to the same camera, just in the different coordinate frames. We make sure this is the case by using the image number to match up camera positions in case some images get deleted from the COLMAP step (which may delete blurry images). With these sets of positions we calculate the homography (which is the 4x4 affine transformation) between the two sets of coordinates. 

Next in the code we create the spiral line that goes around the bounding box. We create this spiral line out of poses, even though we do not need the extra rotation information in the pose. We later just extract the positions from the poses and transform these positions into the model's frame. We also transform the bounding box position into the model's frame and get its center.

Now that we have the bounding box center in the model's frame and the spiral positions along the path in the model's frames, we can calculate the rotation matrix between each individual spiral position and the center of the bounding box. We then use the position and the rotation matrix to a camera pose. We combine all of these poses together to get the camera path. 

## Bounding Box Python File
You will notice that we calculate the bounding box in the model's frame during the process to create the camera path. We use the same logic and functions to create the bounding box in the `get_bounding_box_model_frame.py` file, but we just stop once we create the bounding box. 