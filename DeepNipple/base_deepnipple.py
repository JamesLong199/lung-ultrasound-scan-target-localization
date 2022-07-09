'''
DEEPNIPPLE MAIN SCRIPT
alfonsomedela
alfonmedela@gmail.com
alfonsomedela.com
'''

import argparse
import glob
import pickle
import numpy as np
import cv2

from utils.code.deep_nipple import DeepNipple

from final_phase.subject_info import SUBJECT_NAME, SCAN_POSE

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(description='DeepNipple algorithm')
parser.add_argument('--mode', type=str, default='seg', help='seg or bbox mode')
parser.add_argument('--show', type=bool, default=True, help='show the output')
parser.add_argument('--pose_model', type=str, default='ViTPose_large', help='pose model')


if __name__ == '__main__':

    print('Running DeepNipple...')

    args = parser.parse_args()
    alg_mode = args.mode
    show = args.show
    POSE_MODEL = args.pose_model

    folder_path = 'final_phase/data/' + SUBJECT_NAME + '/' + SCAN_POSE + '/'
    img_root = folder_path + "color_images"
    os.chdir(img_root)
    learner_path = "../../../../../DeepNipple/utils/models/base-model"
    for i, file in enumerate(glob.glob("*.jpg")):
        image_name = os.path.join(file)
        print("image_name: ", image_name)
        output, image = DeepNipple(image_name, learner_path, alg_mode, show)

        cv2.imwrite('../DeepNipple_output_images/cam_{}.jpg'.format(i+1), image)

        # determine left and right nipples
        with open('../' + POSE_MODEL + '/keypoints/cam_{}_keypoints.pickle'.format(i + 1), 'rb') as f:
            pose_results = pickle.load(f)

            if POSE_MODEL == 'OpenPose':
                l_shoulder = np.array(pose_results['people'][0]['pose_keypoints_2d'][15:17])
                r_shoulder = np.array(pose_results['people'][0]['pose_keypoints_2d'][6:8])
            else:
                l_shoulder = np.array(pose_results[0]['keypoints'][5][:2])
                r_shoulder = np.array(pose_results[0]['keypoints'][6][:2])

            l_nipple_idx = np.argmin(np.square(output - l_shoulder).sum(axis=1))
            r_nipple_idx = np.argmin(np.square(output - r_shoulder).sum(axis=1))

            # assert l_nipple_idx != r_nipple_idx

            l_nipple = np.hstack( (output[l_nipple_idx,:], [1]))
            r_nipple = np.hstack( (output[r_nipple_idx,:], [1]))

            if POSE_MODEL == 'OpenPose':
                pose_results['people'][0]['pose_keypoints_2d'] += l_nipple.tolist()
                pose_results['people'][0]['pose_keypoints_2d'] += r_nipple.tolist()
            else:
                pose_results[0]['keypoints'] = np.vstack( (pose_results[0]['keypoints'], l_nipple) )
                pose_results[0]['keypoints'] = np.vstack( (pose_results[0]['keypoints'], r_nipple) )

            # print(pose_results)

        # save nipple coordinates to the file
        with open('../' + POSE_MODEL + '/keypoints/cam_{}_keypoints.pickle'.format(i + 1), 'wb') as f:
            pickle.dump(pose_results, f)



