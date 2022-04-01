# python class for camera calibration object
# assuming checkerboard calibration pattern is used

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class CamCalib:

    # constructor
    # assume global z-coordinate is 0
    def __init__(self,w,h,criteria=(cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
        # inputs
        # -- w: no. of inside corners in x-direction
        # -- h: no. of inside corners in y-direction
        # -- criteria: stopping criteria used in refining subpixel coordinates,
        #              in terms of accuracy and iteration number
        assert isinstance(w,int) and isinstance(h,int)

        self.w = w
        self.h = h
        # 3d corner global coordinates (same for all images)
        self.obj_pts = np.zeros((w*h,3),np.float32)
        # grid of x and y coordinates: (2,w,h) -> (wxh,2)
        # from (0,0,0) to (5,7,0)
        self.obj_pts[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        # list of 2d corner image coordinates (not same for all images)
        self.img_pts_list = []
        # list of 3d corner global coordinates
        self.obj_pts_list = []
        self.criteria = criteria
        # image size of the camera
        self.im_size = None
        # camera intrinsic matrix
        self.cam_matrix = None
        # camera distortion coefficients
        self.dist_coeffs = None

    # add one image containing checkerboard to calibration algorithm
    def add_image(self,fname,plt=False):
        # input
        # -- fname: name of the image file
        # -- plt: if using matplotlib to plot images, change BGR to RGB
        #         default False
        # output
        # -- marked_img: if input image is valid, return annotated image array
        #             return 'None' if input image is invalid

        assert isinstance(fname,str)

        img = cv2.imread(fname)
        # check image shape and pixel value range
        assert len(img.shape) == 2 or len(img.shape) == 3
        assert np.amax(img)<=255 and np.amin(img)>=0

        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        if self.im_size == None:
            self.im_size = gray_img.shape[::-1]
        else:
            # check if all input images are of same size
            assert self.im_size == gray_img.shape[::-1]

        retval,corners = cv2.findChessboardCorners(gray_img,(self.w,self.h),
                         cv2.CALIB_CB_FAST_CHECK)

        # if input image is valid for calibration purpose
        if retval == True:
            # obtain refined sub-pixel 2d image coordinates
            corners_r = cv2.cornerSubPix(gray_img,corners,
                        (11,11),(-1,-1),self.criteria)
            self.img_pts_list.append(corners_r)
            self.obj_pts_list.append(self.obj_pts)

        else:
            print('Invalid input image!')
            return None

        marked_img = cv2.drawChessboardCorners(img,(self.w,self.h),
                     corners_r,patternWasFound=True)

        if plt == True and len(marked_img.shape) == 3:
            return np.flip(marked_img,axis=2)

        else:
            return marked_img


    # camera calibration
    def run_calibration(self):
        # need at least 3 images to calibrate
        assert len(self.obj_pts_list) >= 3

        retval, cameraMatrix, distCoeffs, rvecs, tvecs = \
        cv2.calibrateCamera(self.obj_pts_list,self.img_pts_list,
        self.im_size,None,None)

        self.cam_matrix = cameraMatrix
        self.dist_coeffs = distCoeffs

        return retval, cameraMatrix, distCoeffs, rvecs, tvecs


    # image undistortion
    # can only run it when camera is calibrated
    def run_undistortion(self,fname,plt=False):
        # input
        # -- fname: name of the image file
        # -- plt: if using matplotlib to plot images, change BGR to RGB
        #         default False
        # output
        # -- img_re: restored image
        assert isinstance(fname,str)
        assert self.cam_matrix is not None
        assert self.dist_coeffs is not None

        img = cv2.imread(fname)
        # check image shape and pixel value range
        assert len(img.shape) == 2 or len(img.shape) == 3
        assert np.amax(img)<=255 and np.amin(img)>=0

        h,w = img.shape[:2]
        # refined camera matrix that eliminates undistortion
        # region of interest that can be used to crop result
        new_cam_mtx,roi = cv2.getOptimalNewCameraMatrix(self.cam_matrix,self.dist_coeffs,(w,h),1,(w,h))

        # undistort image
        img_re = cv2.undistort(img, self.cam_matrix, self.dist_coeffs, None, new_cam_mtx)

        # crop the image
        x,y,m,n = roi
        img_re = img_re[y:y+n, x:x+m]

        if plt == True and len(img_re.shape) == 3:
            return np.flip(img_re,axis=2)

        else:
            return img_re
