# convert json keypoints to pickle keypoints, only for OpenPose

import json
import pickle
import os
import argparse

from subject_info import SUBJECT_NAME, SCAN_POSE

parser = argparse.ArgumentParser(description='json2pickle')
parser.add_argument('--subject_name', type=str, default='John Doe', help='subject name')
parser.add_argument('--scan_pose', type=str, default='none', help='scan pose')
args = parser.parse_args()
if args.subject_name != 'John Doe':
    SUBJECT_NAME = args.subject_name
if args.scan_pose != 'none':
    SCAN_POSE = args.scan_pose

folder_path = 'main_project/data/' + SUBJECT_NAME + '/' + SCAN_POSE + '/' + 'OpenPose/keypoints/'

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


