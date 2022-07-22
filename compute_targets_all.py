# Compute 3 targets for all subjects using 3 HPE models

import os
import subprocess

if __name__ == '__main__':

    data_path = 'final_phase/data'

    # loop through the data folder
    for SUBJECT_NAME in os.listdir(data_path):
        print("subject: ", SUBJECT_NAME)
        subject_folder_path = os.path.join(data_path, SUBJECT_NAME)
        if os.path.isfile(subject_folder_path):
            continue

        # loop through two poses (front and side)
        for SCAN_POSE in os.listdir(subject_folder_path):
            print("scan pose: ", SCAN_POSE)

            # ViTPose_base
            subprocess.run([
                "python", "ViTPose/demo/top_down_img_demo_with_mmdet.py",
                "ViTPose/demo/mmdetection_cfg/yolov3_d53_320_273e_coco.py",
                "ViTPose/models/yolov3_d53_320_273e_coco.pth",
                "ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_base_coco_256x192.py",
                "ViTPose/models/vitpose-b.pth",
                "--subject_name={}".format(SUBJECT_NAME),
                "--scan_pose={}".format(SCAN_POSE),
                "--pose_model=ViTPose_base"
            ])

            # OpenPose
            image_dir = 'final_phase/data/{}/{}/color_images'.format(SUBJECT_NAME, SCAN_POSE)
            write_images = 'final_phase/data/{}/{}/OpenPose/output_images/'.format(SUBJECT_NAME, SCAN_POSE)
            write_json = 'final_phase/data/{}/{}/OpenPose/keypoints/'.format(SUBJECT_NAME, SCAN_POSE)
            subprocess.run([
                "python", "final_phase/openpose_python.py",
                "--image_dir", image_dir, "--write_images", write_images, "--write_json", write_json
            ])
            subprocess.run([
                "python", "final_phase/json2pickle.py",
                "--subject_name={}".format(SUBJECT_NAME),
                "--scan_pose={}".format(SCAN_POSE)
            ])

            # loop through three HPE models to compute targets
            for POSE_MODEL in ['OpenPose', 'ViTPose_base', 'ViTPose_large']:
                print("pose model: ", POSE_MODEL)
                subprocess.run([
                    "python", "final_phase/compute_target.py",
                    "--pose_model={}".format(POSE_MODEL),
                    "--subject_name={}".format(SUBJECT_NAME),
                    "--scan_pose={}".format(SCAN_POSE),
                ])

