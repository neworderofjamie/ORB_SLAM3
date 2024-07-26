import cv2
import numpy as np
import os
import sys

from scipy.spatial.transform import Rotation

from glob import glob

assert len(sys.argv) > 1

out_size = int(sys.argv[2] if len(sys.argv) > 2 else 512)
out_fov_degree = float(sys.argv[3] if len(sys.argv) > 3 else 120.0)

# Load calibration data
calibration = np.load("calibration.npz")
omni_k = calibration["k"]
omni_xi = calibration["xi"]
omni_d = calibration["d"]

rotation = Rotation.from_euler("zx", [180.0, 30], degrees=True).as_matrix()

out_fov_pixels = (out_size / 2.0) / np.tan(np.radians(out_fov_degree / 2.0))
k_out = np.asarray([[out_fov_pixels,  0,                out_size / 2],
                    [0,               out_fov_pixels,   out_size / 2],
                    [0,               0,                1]])


print(f"Fx={k_out[0,0]}, Fy={k_out[1,1]}, Cx={k_out[0,2]}, Cx={k_out[1,2]}")

# Loop through files
for f in glob(os.path.join(sys.argv[1], "*.jpg")):
    # Split up path
    path, filename = os.path.split(f)
    title, ext = os.path.splitext(filename)
    
    # Read omni directional image
    omni_img = cv2.imread(f)
    
    # Convert to perspective
    persp_img = cv2.omnidir.undistortImage(omni_img, omni_k, omni_d, omni_xi, cv2.omnidir.RECTIFY_PERSPECTIVE,
                                           Knew=k_out, new_size=(out_size, out_size), R=rotation)
    
    # Write back
    cv2.imwrite(os.path.join(path, f"{title}_persp.{ext}"), persp_img)
