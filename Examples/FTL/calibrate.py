import cv2
import numpy as np
from scipy.spatial.transform import Rotation

def build_map(w, h, r, offset, centre_x, centre_y):
    x = np.arange(0, w)
    y = np.arange(0, h)
    
    rs = r * y / h
    
    thetas = np.expand_dims(((x - offset) / w) * 2 * np.pi, 1)

    map_x = np.transpose(centre_x - rs * np.sin(thetas)).astype(np.float32)
    map_y = np.transpose(centre_y - rs * np.cos(thetas)).astype(np.float32)
    return map_x, map_y

IN_W = 1440
IN_H = 1440
FOV = 120.0

CHESS_COLS = 9
CHESS_ROWS = 6
CHESS_SIZE_MM = 43.5

CALIBRATE = False
NUM_CALIBRATE_POINTS = 100

vid = cv2.VideoCapture(2)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, IN_W)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, IN_H)

if CALIBRATE:
    sub_pix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    pattern_points = np.zeros((1, CHESS_COLS * CHESS_ROWS, 3), dtype=np.float32)
    pattern_points[0, :, :2] = np.indices((CHESS_COLS, CHESS_ROWS)).T.reshape(-1, 2)
    pattern_points *= CHESS_SIZE_MM

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    index = 0
    while True:
        # Read frame
        ret, frame = vid.read()
        
        # Unwrap
        #unwrapped = cv2.remap(frame, xmap, ymap, cv2.INTER_NEAREST)
        
        # Crop out FOV
        #frame = unwrapped[:,fov_crop:-fov_crop]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_copy = np.copy(frame)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (CHESS_COLS, CHESS_ROWS), None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1), sub_pix_criteria)
            cv2.drawChessboardCorners(frame_copy, (CHESS_COLS, CHESS_ROWS), corners2, ret)
            
            objpoints.append(pattern_points)
            imgpoints.append(corners2)
        
        cv2.imshow("unwrapped", frame_copy)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord("s"):
            cv2.imwrite(f"test{index}.png", frame)
            index += 1

    h, w = frame.shape[:2]

    print(f"{len(objpoints)} calibration points captured")

    # Subsample calibration points for speed
    inds = np.random.choice(len(objpoints), NUM_CALIBRATE_POINTS)
    objpoints = np.asarray(objpoints)[inds]
    imgpoints = np.asarray(imgpoints)[inds]

    calibration_flags = cv2.omnidir.CALIB_USE_GUESS + cv2.omnidir.CALIB_FIX_SKEW + cv2.omnidir.CALIB_FIX_CENTER
    rms, omni_k, omni_xi, omni_d, _, _, _ = cv2.omnidir.calibrate(
        objpoints, imgpoints, gray.shape[::-1],
        K=None, xi=None, D=None,
        flags=calibration_flags, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8))
    
    print("Done calibrating")
    print(omni_k, omni_xi, omni_d)
    np.savez("calibration.npz", k=omni_k, xi=omni_xi, d=omni_d)
else:
    calibration = np.load("calibration.npz")
    omni_k = calibration["k"]
    omni_xi = calibration["xi"]
    omni_d = calibration["d"]
    """
    omni_k = np.asarray([[422.05218858, 0.0,            720.0],
                         [0.0,          428.31123944,   720.0],
                         [0.0,          0.0,            1.0]])
    omni_xi = np.asarray([[0.39416013]])
    omni_d = np.asarray([[-0.12039939, 0.00880851, 0.00033161, 0.00195181]])
    """

OUT_SIZE = (512, 512)
K_OUT = np.asarray([[OUT_SIZE[0] / 4,   0,                  OUT_SIZE[0] / 2],
                    [0,                 OUT_SIZE[1] / 4,    OUT_SIZE[1] / 2],
                    [0,                 0,                  1]])
while True:
    # Read frame
    ret, frame = vid.read()
    
    # Unwrap
    #unwrapped = cv2.remap(frame, xmap, ymap, cv2.INTER_NEAREST)
    
    # Crop out FOV and top
    #frame = unwrapped[:,fov_crop:-fov_crop]
    
    # undistort
    #dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    #dst_pin = cv2.remap(frame, pin_x_map, pin_y_map, cv2.INTER_NEAREST)
    #dst_fish = cv2.remap(frame, fish_x_map, fish_y_map, cv2.INTER_NEAREST)
    #rotation = Rotation.from_euler(axes[axis], angle, degrees=True)
    rotation = Rotation.from_euler("zx", [180.0, 30], degrees=True)
    dst_omni = cv2.omnidir.undistortImage(frame, omni_k, omni_d, omni_xi, cv2.omnidir.RECTIFY_PERSPECTIVE,
                                          Knew=K_OUT, new_size=OUT_SIZE, R=rotation.as_matrix())
    
    #cv2.putText(dst_omni, f"{axes[axis]} {angle}", (0,20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,0,0))
    cv2.imshow("unwrapped", frame)
    #cv2.imshow("pinhole undistort", dst_pin)
    cv2.imshow("omni undistort", dst_omni)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    
