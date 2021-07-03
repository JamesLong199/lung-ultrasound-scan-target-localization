import cv2
import pupil_apriltags
import numpy as np

# python class for camera localization from AprilTag images
# for apriltag detector, the only mandatory input is tag family
# can add a dictionary of optional params for apriltag as constructor input later
class CamPoseDetector:
    # constructor
    def __init__(self,cam_matrix,dist_coeffs,tag_fam):
        # inputs
        # -- cam_matrix: instrinsic matrix of camera
        # -- dist_coeffs: distortion coefficients of camera
        # -- tag_fam: tag families to be detected, a string separated by spaces
        assert cam_matrix.shape == (3,3)
        assert dist_coeffs.shape == (1,5)
        assert isinstance(tag_fam,str)

        self.cam_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs
        self.detector = pupil_apriltags.Detector(families=tag_fam)

    def run_undistortion(self,img):
        # inputs
        # -- img: the image to be undistorted
        # outputs
        # -- img_re: restored image
        h,w = img.shape[:2]
        # refined camera matrix that eliminates undistortion
        # region of interest that can be used to crop result
        new_cam_mtx,roi = cv2.getOptimalNewCameraMatrix(self.cam_matrix,self.dist_coeffs,(w,h),1,(w,h))
        # undistort image
        img_re = cv2.undistort(img, self.cam_matrix, self.dist_coeffs, None, new_cam_mtx)

        return img_re,new_cam_mtx

    def get_camera_pose(self,img,tag_size,verbose=False):
        # inputs
        # -- img: image containing Apriltag
        # -- tag_size: tag size in meter, must not be None
        # -- verbose: print all detection parameters if True
        # outputs
        # -- cam_pose: list of camera pose in tag frame for all tags detected
        assert isinstance(img,np.ndarray)
        assert isinstance(tag_size,int) or isinstance(tag_size,float)

        # check image shape and pixel value range
        assert len(img.shape) == 2 or len(img.shape) == 3
        assert np.amax(img)<=255 and np.amin(img)>=0
        # undistort camera image and convert it to grayscale
        # required for apriltag detection
        img_re,new_cam_mtx = self.run_undistortion(img)
        # img_re,new_cam_mtx = img, self.cam_matrix
        gray_img = cv2.cvtColor(img_re,cv2.COLOR_BGR2GRAY)

        # apriltag detection and pose estimation
        # changed second param to negative to get valid axis orientation
        intrinsics = [new_cam_mtx[0,0],new_cam_mtx[1,1],new_cam_mtx[0,2],
                 new_cam_mtx[1,2]]

        detection = self.detector.detect(gray_img,estimate_tag_pose=True,
        camera_params=intrinsics,tag_size=tag_size)

        if verbose == True and detection is not None:
            print('{} tags detected'.format(len(detection)))
            print(detection)

        return detection
