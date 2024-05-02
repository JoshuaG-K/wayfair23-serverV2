[`ply_cull.py`](/public/scripts/ply_cull.py) is a Python script that can crop the ply output from 3D Gaussian Splatting to a bounding box in under a minute. To run it, simply call
```bash
python ply_cull.py GAUSSIAN_TRAINING_OUTPUT.ply -b BOUNDING_BOX.json -o OUTPUT_LOCATION.ply
```

The bounding box json file should be of the following format:
- Each value is an array of 3 numbers representing the xyz coordinate
- The keys are `bot_left_front`, `top_left_front`, `top_left_back`, `top_right_back`, `bot_left_back`, `bot_right_front`, `top_right_front`, `bot_right_back`, and `center`, representing the 8 corners of the bounding box and is center respectively
The `readBounds()` function can be easily modified to accept different formats.

The script uses [Open3D](https://www.open3d.org/) libraries to construct a `o3d.geometry.OrientedBoundingBox` object and uses that to select the indices of points to be culled. Currently, Open3D point-cloud functions cannot properly modify the `.ply` format used by 3D Gaussian Splatting, so we rely on the [Plyfile](https://python-plyfile.readthedocs.io/en/latest/index.html) library functions to perform the actual point culling.

# References
- https://www.open3d.org/
- https://python-plyfile.readthedocs.io/en/latest/index.html
- https://www.open3d.org/docs/release/python_api/open3d.geometry.OrientedBoundingBox.html#open3d.geometry.OrientedBoundingBox.create_from_points
- https://stackoverflow.com/questions/71051649/select-remaining-points-after-cropping-a-point-cloud

# Known Issues
`o3d.geometry.OrientedBoundingBox.create_from_points()` is used to construct the bounding box information from the json file. It seems that this method is not perfectly accurate, and could possibly be replaced with a more accurate method.
