# Run `ViTPose_UR_move.py`. Construct 3D point cloud, compute target normal/orientation, move the robot arm.

import subprocess
from final_phase.subject_info import SUBJECT_NAME, SCAN_POSE

POSE_MODEL = 'OpenPose'  # ViTPose_large, ViTPose_base, OpenPose

if __name__ == '__main__':
    # # Capture two color/depth images with two Intel Realsense depth camera
    # subprocess.run([
    #                 "python", "final_phase/ViTPose_UR_collect_data.py",
    #                 ])
    #
    #
    # Run pose estimation on the two color images, save shoulders & hips keypoints.
    # ViTPose large:
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
        image_dir = 'final_phase/data/{}/{}/color_images'.format(SUBJECT_NAME, SCAN_POSE)
        write_images = 'final_phase/data/{}/{}/OpenPose/output_images/'.format(SUBJECT_NAME, SCAN_POSE)
        write_json = 'final_phase/data/{}/{}/OpenPose/keypoints/'.format(SUBJECT_NAME, SCAN_POSE)

        subprocess.run([
            "python", "final_phase/openpose_python.py",
            "--image_dir", image_dir, "--write_images", write_images, "--write_json", write_json
        ])

        subprocess.run([
            "python", "final_phase/json2pickle.py"
        ])
    #
    #
    # # nipple detection
    # # run DeepNipple on the two color images, save nipple keypoints.
    # subprocess.run([
    #                 "python", "DeepNipple/base_deepnipple.py",
    #                 "--mode=bbox",
    #                 "--show=True",
    #                 "--pose_model={}".format(POSE_MODEL)
    #                 ])

    # Compute target coordinates, and save them
    subprocess.run([
                    "python", "final_phase/compute_target.py",
                    "--pose_model={}".format(POSE_MODEL)
                    ])

    # # Move robot
    # subprocess.run([
    #                 "python", "final_phase/ViTPose_UR_move.py",
    #                 "--pose_model={}".format(POSE_MODEL)
    #                ])
