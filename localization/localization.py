from CamPoseDetector import CamPoseDetector
from TagList import TagList
from pose_conversion import transformation_to_pose,pose_to_transformation,eulerAnglesToRotationMatrix,rotationMatrixToEulerAngles
import math
import numpy as np
import pickle
import cv2
import pupil_apriltags


# assume uniform tag_size for all tags used
def localize_camera(fname,cam_matrix,dist_coeffs,tag_list):
    # inputs
    # -- fname: file path of image
    # -- cam_matrix: instrinsic matrix of camera
    # -- dist_coeffs: distortion coefficients of camera
    # -- tag_fam: tag families to be detected, a string separated by spaces
    # -- tag_size: tag size in meter, must not be None
    # output
    # -- (avg_pos,avg_angles): average pos and angles(degree) of camera from all
    #                          tag detections in the image
    assert cam_matrix.shape == (3,3)
    assert dist_coeffs.shape == (1,5)
    assert isinstance(tag_list,TagList)
    assert isinstance(fname,str)

    # obtain camera pose from image
    tag_families = tag_list.get_tag_families()
    tag_size = tag_list.get_tag_size()
    myCPD = CamPoseDetector(cam_matrix,dist_coeffs,tag_families)
    det_list = myCPD.get_camera_pose(fname,tag_size=tag_size,verbose=False)

    # for each tag detection, compute global position and orientation of camera
    avg_pos = np.zeros((3,))
    avg_angles = np.zeros((3,))

    for det in det_list:
        cam_R = det.pose_R
        cam_t = (det.pose_t).reshape(3,)
        # camera position in tag frame
        cam_pos_tag = cam_R.T @ (-cam_t)

        tag_pose = tag_list.get_tag_pose(det.tag_family.decode('utf-8'),det.tag_id)
        tag_R,tag_t = pose_to_transformation(tag_pose)

        # camera global position
        cam_pos_global = tag_R @ cam_pos_tag + tag_t

        # camera global angles
        total_R = cam_R @ tag_R
        camera_angles_global = rotationMatrixToEulerAngles(total_R)
        avg_pos += cam_pos_global
        avg_angles += camera_angles_global

        # debug
        # camera_angles_tag = rotationMatrixToEulerAngles(cam_R)* 180 / math.pi
        # camera_angles_offset = rotationMatrixToEulerAngles(tag_R)* 180 / math.pi
        # print('camera_angles_tag:',camera_angles_tag)
        # print('camera_angles_offset:',camera_angles_offset)

    avg_pos = avg_pos / len(det_list)
    avg_angles = avg_angles / len(det_list)
    # convert angles to degrees
    avg_angles = avg_angles * 180 / math.pi

    return (avg_pos,avg_angles)



if __name__ == '__main__':

    with open('cam_params.pickle', 'rb') as f:
        cam_matrix,dist_coeffs = pickle.load(f)

    # img_dir = '/home/james/projects/localization/changing_translation'
    img_dir = '/home/james/projects/localization/changing_orientation'

    fname = img_dir + '/' + '-60_54.6.jpg'

    # R_t2g = np.array([[1,0,0],[0,0,1],[0,-1,0]]) # corresponds to (-90,0,0) euler angles
    myTagList = TagList(tag_size=0.084)
    myTagList.add_tag(family='tagStandard41h12',id=0,pos=(0.1016,0,1.395),
            angles=(-90,0,0),radian=False)

    cam_pos,cam_angles = localize_camera(fname,cam_matrix,dist_coeffs,myTagList)

    np.set_printoptions(precision=3)
    print('camera global position:\n{}'.format(cam_pos))
    print('camera angles:\n{}'.format(cam_angles))
