# convert json keypoints to pickle keypoints, only for OpenPose

import json
import pickle
import os

from subject_info import SUBJECT_NAME, SCAN_POSE

folder_path = 'final_phase/data/' + SUBJECT_NAME + '/' + SCAN_POSE + '/' + 'OpenPose/keypoints/'

with open(folder_path + 'cam_1_keypoints.json') as f:
    cam1_data = json.load(f)
    with open(folder_path + 'cam_1_keypoints.pickle', 'wb') as f:
        pickle.dump(cam1_data, f)

with open(folder_path + 'cam_2_keypoints.json') as f:
    cam2_data = json.load(f)
    with open(folder_path + 'cam_2_keypoints.pickle', 'wb') as f:
        pickle.dump(cam2_data, f)

os.remove(folder_path + 'cam_1_keypoints.json')
os.remove(folder_path + 'cam_2_keypoints.json')


