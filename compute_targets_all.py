# Compute 3 targets for all subjects using 3 HPE models

import os
import subprocess

# ratios = {
#     'ViTPose_large': {'target1': [0.29194393, 0.12326672], 'target2': [0.27588455, 0.57679719], 'target4': [0.3438, 0.1083]},
#     'ViTPose_base': {'target1': [0.29466347, 0.12381275], 'target2': [0.28210566, 0.583985], 'target4': [0.3536, 0.0894]},
#     'OpenPose': {'target1': [0.29612489, 0.11958833], 'target2': [0.29058278, 0.57921349], 'target4': [0.3420, 0.0516]}
# }

ratios = {
    'ViTPose_large': {'target1': [0.3, 0.1], 'target2': [0.3, 0.55], 'target4': [0.35, 0.1]},
    'ViTPose_base': {'target1': [0.3, 0.1], 'target2': [0.3, 0.55], 'target4': [0.35, 0.1]},
    'OpenPose': {'target1': [0.3, 0.1], 'target2': [0.3, 0.55], 'target4': [0.35, 0.1]}
}


if __name__ == '__main__':

    data_path = 'final_phase/data'

    # loop through the data folder
    for SUBJECT_NAME in os.listdir(data_path):
        if SUBJECT_NAME != 'henry_kuang':
            continue

        print("subject: ", SUBJECT_NAME)
        subject_folder_path = os.path.join(data_path, SUBJECT_NAME)
        if os.path.isfile(subject_folder_path):
            continue

        # loop through two poses (front and side)
        for SCAN_POSE in os.listdir(subject_folder_path):
            if SCAN_POSE != 'side':
                continue

            print("scan pose: ", SCAN_POSE)

            # loop through three HPE models to compute targets
            for POSE_MODEL in ['OpenPose', 'ViTPose_base', 'ViTPose_large']:
            # for POSE_MODEL in ['ViTPose_large']:
                print("pose model: ", POSE_MODEL)
                subprocess.run([
                    "python", "final_phase/compute_target.py",
                    "--pose_model={}".format(POSE_MODEL),
                    "--subject_name={}".format(SUBJECT_NAME),
                    "--scan_pose={}".format(SCAN_POSE),
                    "--target1_r1", str(ratios[POSE_MODEL]['target1'][0]),
                    "--target1_r2", str(ratios[POSE_MODEL]['target1'][1]),
                    "--target2_r1", str(ratios[POSE_MODEL]['target2'][0]),
                    "--target2_r2", str(ratios[POSE_MODEL]['target2'][1]),
                    "--target4_r1", str(ratios[POSE_MODEL]['target4'][0]),
                    "--target4_r2", str(ratios[POSE_MODEL]['target4'][1])
                ])

        # break