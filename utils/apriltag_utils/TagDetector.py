import cv2 as cv
import pupil_apriltags
import numpy as np


# python class for camera localization from AprilTag images

class TagDetector:
    def __init__(self, cam_matrix, dist_coeffs, tag_family, cam_type):
        """
        Constructor for a TagDetector object
        :param cam_matrix: camera intrinsic matrix
        :param tag_fam: tag family
        :param cam_type: the camera type: "standard" or "fisheye"
        :param dist_coeffs: distortion coefficients of camera, could be None
        """
        assert cam_matrix.shape == (3, 3)
        if dist_coeffs is not None:
            assert dist_coeffs.shape == (1, 5)
            self.dist_coeffs = dist_coeffs
        assert isinstance(tag_family, str)
        assert isinstance(cam_type, str)

        self.cam_matrix = cam_matrix
        self.cam_type = cam_type
        self.detector = pupil_apriltags.Detector(families=tag_family)

    def undistort_image_standard(self, img):
        """
        Standard undistorition
        :param img: the original image with distortion
        :return: the undistorted version of the camera matrix and image
        """
        h, w = img.shape[:2]
        # refined camera matrix that eliminates undistortion
        # region of interest that can be used to crop result
        new_cam_matrix, roi = cv.getOptimalNewCameraMatrix(self.cam_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        # undistort image
        img_re = cv.undistort(img, self.cam_matrix, self.dist_coeffs, None, new_cam_matrix)

        return img_re, new_cam_matrix

    def undistort_image_fisheye(self, img, balance=1.0, DIM=(1280, 720), dim2=None, dim3=None):
        """
        Fisheye undistortion
        :param img: the original image with distortion
        :param balance:
        :param DIM:
        :param dim2:
        :param dim3:
        :return: the undistorted version of the camera matrix and image
        """
        K, D = self.cam_matrix, self.dist_coeffs
        dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
        assert dim1[0] / dim1[1] == DIM[0] / DIM[
            1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
        if not dim2:
            dim2 = dim1
        if not dim3:
            dim3 = dim1
        scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image.
        # OpenCV document failed to make this clear!
        new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
        map1, map2 = cv.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv.CV_16SC2)
        undistorted_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

        return undistorted_img, new_K

    def detect_tags(self, img, tag_size, verbose=False):
        """
        Detect all Apriltags in the input image
        :param img: the input image
        :param tag_size: tag size in meters
        :param verbose: print all detection parameters if True
        :return: the undistorted version of the image and the detection results
        """

        assert isinstance(img, np.ndarray)
        assert isinstance(tag_size, int) or isinstance(tag_size, float)

        # check image shape and pixel value range
        assert len(img.shape) == 2 or len(img.shape) == 3
        assert np.amax(img) <= 255 and np.amin(img) >= 0

        # undistort camera image and convert it to grayscale
        # required for apriltag detection
        if self.dist_coeffs is not None:
            undistorted_img, new_cam_mat = None, None
            if self.cam_type == "fisheye":
                undistorted_img, new_cam_mat = self.undistort_image_fisheye(img)
            elif self.cam_type == "standard":
                undistorted_img, new_cam_mat = self.undistort_image_standard(img)
        else:
            undistorted_img = img
            new_cam_mat = self.cam_matrix

        gray_img = cv.cvtColor(undistorted_img, cv.COLOR_BGR2GRAY)

        # apriltag detection and pose estimation
        # changed second param to negative to get valid axis orientation
        intrinsics = [new_cam_mat[0, 0], new_cam_mat[1, 1], new_cam_mat[0, 2],
                      new_cam_mat[1, 2]]

        detection = self.detector.detect(gray_img, estimate_tag_pose=True,
                                         camera_params=intrinsics, tag_size=tag_size)

        if verbose == True and detection is not None:
            print('{} tags detected'.format(len(detection)))
            print(detection)

        return undistorted_img, detection
