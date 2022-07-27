# The entire pipeline of one experiment trial

import subprocess
from main_project.subject_info import SUBJECT_NAME, SCAN_POSE

POSE_MODEL = 'OpenPose'  # ViTPose_large, ViTPose_base, OpenPose

if __name__ == '__main__':
    # Capture two color/depth images with two Intel Realsense depth cameras
    subprocess.run([
                    "python", "main_project/collect_data.py",
                    ])

    # Run human pose estimation on the two color images, save shoulders & hips keypoints.
    if POSE_MODEL == 'ViTPose_large':
        subprocess.run([
                        "python", "ViTPose/demo/top_down_img_demo_with_mmdet.py",
                        "ViTPose/demo/mmdetection_cfg/yolov3_d53_320_273e_coco.py",
                        "ViTPose/models/yolov3_d53_320_273e_coco.pth",
                        "ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py",
                        "ViTPose/models/vitpose-l-multi-coco.pth",
                        "--subject_name={}".format(SUBJECT_NAME),
                        "--scan_pose={}".format(SCAN_POSE),
                        "--pose_model=ViTPose_large"
                        ])

    elif POSE_MODEL == 'ViTPose_base':
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

    else:
        image_dir = 'main_project/data/{}/{}/color_images'.format(SUBJECT_NAME, SCAN_POSE)
        write_images = 'main_project/data/{}/{}/OpenPose/output_images/'.format(SUBJECT_NAME, SCAN_POSE)
        write_json = 'main_project/data/{}/{}/OpenPose/keypoints/'.format(SUBJECT_NAME, SCAN_POSE)

        subprocess.run([
            "python", "main_project/openpose_python.py",
            "--image_dir", image_dir, "--write_images", write_images, "--write_json", write_json
        ])

        subprocess.run([
            "python", "main_project/json2pickle.py"
        ])

    # Compute target coordinates, and save them
    subprocess.run([
                    "python", "main_project/compute_target.py",
                    "--pose_model={}".format(POSE_MODEL)
                    ])

    # Move robot
    subprocess.run([
                    "python", "main_project/UR_move.py",
                    "--pose_model={}".format(POSE_MODEL)
                   ])
