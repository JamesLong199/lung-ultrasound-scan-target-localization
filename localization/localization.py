from CamPoseDetector import CamPoseDetector
from TagList import TagList
from pose_conversion import transformation_to_pose,pose_to_transformation,eulerAnglesToRotationMatrix,rotationMatrixToEulerAngles
import math
import numpy as np
import pickle
import cv2
import pupil_apriltags
import pandas as pd
import os
from tabulate import tabulate


# assume uniform tag_size for all tags used
def localize_camera(fname,cam_matrix,dist_coeffs,tag_list):
    # inputs
    # -- fname: file path of image
    # -- cam_matrix: instrinsic matrix of camera
    # -- dist_coeffs: distortion coefficients of camera
    # -- tag_list: TagList object
    # output
    # -- (pos,angles): vertically stacked results of all tags' detections
    assert cam_matrix.shape == (3,3)
    assert dist_coeffs.shape == (1,5)
    assert isinstance(tag_list,TagList)
    assert isinstance(fname,str)

    # obtain camera pose from image
    tag_families = tag_list.get_tag_families()
    tag_size = tag_list.get_tag_size()
    myCPD = CamPoseDetector(cam_matrix,dist_coeffs,tag_families)
    img = cv2.imread(fname)
    det_list = myCPD.get_camera_pose(img,tag_size=tag_size,verbose=False)

    # for each tag detection, compute global position and orientation of camera
    avg_pos = np.zeros((3,))
    avg_angles = np.zeros((3,))

    pos = []
    angles = []

    for det in det_list:
        # transformation from tag frame to camera frame
        R_tag_cam = det.pose_R
        t_tag_cam = (det.pose_t).reshape(3,)
        # camera position in tag frame
        cam_pos_tag = R_tag_cam.T @ (-t_tag_cam)

        tag_pose_world = tag_list.get_tag_pose(det.tag_family.decode('utf-8'),det.tag_id)
        R_tag_world,t_tag_world = pose_to_transformation(tag_pose_world)

        # camera global position
        cam_pos_global = R_tag_world @ cam_pos_tag + t_tag_world

        # camera global angles
        R_cam_world = R_tag_world @ R_tag_cam.T
        camera_angles_global = rotationMatrixToEulerAngles(R_cam_world) * 180 / math.pi

        pos.append(cam_pos_global)
        angles.append(camera_angles_global)


    # convert angles to degrees
    pos = np.vstack(pos)
    angles = np.vstack(angles)

    return (pos,angles)



# function that group experiment results together
def group_detection_result(img_dir,cam_matrix,dist_coeffs,tag_list,display_all=False):
    # input
    # -- img_dir: directory of images containing AprilTag
    # -- cam_matrix: instrinsic matrix of camera
    # -- dist_coeffs: distortion coefficients of camera
    # -- tag_list: TagList object
    # -- display_all: if display all detection results from individual tags,
    #                 default False
    # output
    # -- df_result: pandas dataframe of results from all images in the directory
    assert isinstance(img_dir,str)

    if display_all == False:
        result_dict = {'img_name':[],'# tags':[],'avg_pos':[],'std_pos':[],'avg_angles':[],'std_angles':[]}
    else:
        result_dict = {'img_name':[],'# tags':[],'pos':[],'avg_pos':[],'std_pos':[],'angles':[],'avg_angles':[],'std_angles':[]}

    for i,img_name in enumerate(sorted(os.listdir(img_dir))):
        fname = img_dir + '/' + img_name
        pos,angles = localize_camera(fname,cam_matrix,dist_coeffs,tag_list)

        avg_pos = np.mean(pos,axis=0)
        avg_angles = np.mean(angles,axis=0)
        if pos.size >= 3:
            std_pos = np.std(pos,axis=0)
            std_angles = np.std(angles,axis=0)

        result_dict['img_name'].append(img_name)
        result_dict['avg_pos'].append(avg_pos)
        result_dict['std_pos'].append(std_pos)
        result_dict['avg_angles'].append(avg_angles)
        result_dict['std_angles'].append(std_angles)
        result_dict['# tags'].append((pos.size/3))

        if display_all:
            result_dict['pos'].append(pos)
            result_dict['angles'].append(angles)

    df_result = pd.DataFrame(result_dict)
    return df_result





if __name__ == '__main__':

    with open('cam_params.pickle', 'rb') as f:
        cam_matrix,dist_coeffs = pickle.load(f)

    np.set_printoptions(precision=3)

    # R_t2g = np.array([[1,0,0],[0,0,1],[0,-1,0]]) # corresponds to (-90,0,0) euler angles

    # tag36h11
    myTagList = TagList(tag_size=0.048)
    myTagList.add_tag(family='tag36h11',id=0,pos=(0.04,0,1.185),
            angles=(-90,0,0),radian=False)
    myTagList.add_tag(family='tag36h11',id=1,pos=(0.04,0,1.519),
            angles=(-90,0,0),radian=False)
    myTagList.add_tag(family='tag36h11',id=2,pos=(0.233,0,1.519),
            angles=(-90,0,0),radian=False)
    # img_dir = '/home/james/projects/localization/mannequin_pics'
    img_dir = '/home/james/projects/localization/vertical_pics'
    # img_dir = '/home/james/projects/localization/horizontal_and_close'

    # tag41h12
    # myTagList = TagList(tag_size=0.084)
    # myTagList.add_tag(family='tagStandard41h12',id=0,pos=(0.1016,0,1.395),
    #         angles=(-90,0,0),radian=False)
    # img_dir = '/home/james/projects/localization/changing_translation'
    # img_dir = '/home/james/projects/localization/changing_orientation'


    df_result = group_detection_result(img_dir,cam_matrix,dist_coeffs,myTagList,display_all=False)
    # displaying the DataFrame
    print(tabulate(df_result, headers = 'keys', tablefmt = 'fancy_grid'))
