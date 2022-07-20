import os
import shutil
import pickle

SUBJECT_NAME = 'benny_cai'

SCAN_POSE = 'side'
# SCAN_POSE = 'front'

if __name__ == '__main__':

    folder_path = 'data/' + SUBJECT_NAME
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    os.chdir(folder_path)

    if not os.path.isdir(SCAN_POSE):
        os.mkdir(SCAN_POSE)
    os.chdir(SCAN_POSE)

    with open('ground_truth.pickle', 'wb') as f:
        if SCAN_POSE == 'front':
            pickle.dump({'target_1': None, 'target_2': None}, f)
        else:
            pickle.dump({'target_4': None}, f)

    for subfolder in ['color_images',
                      'DeepNipple_output_images',
                      'depth_images',
                      'intrinsics',
                      'OpenPose',
                      'ViTPose_base',
                      'ViTPose_large']:
        if not os.path.isdir(subfolder):
            os.mkdir(subfolder)

    # copy odometry.log from 'data' folder
    src = '../../odometry.log'
    dst = 'odometry.log'
    shutil.copyfile(src, dst)

    file1 = open("../../subject_name.txt", "a")
    file1.write(SUBJECT_NAME + '_' + SCAN_POSE + "\n")
    file1.close()

    for subfolder in ['OpenPose',
                      'ViTPose_base',
                      'ViTPose_large']:

        for subsubfolder in ['keypoints', 'output_images']:
            if not os.path.isdir(subfolder + '/' + subsubfolder):
                os.mkdir(subfolder + '/' + subsubfolder)







