# Run `ViTPose_UR_move.py`. Construct 3D point cloud, compute target normal/orientation, move the robot arm.


import subprocess
from final_phase.subject_info import SUBJECT_NAME, SCAN_POSE

# # Capture two color/depth images with two Intel Realsense depth camera
# subprocess.run([
#                 "python", "final_phase/ViTPose_UR_collect_data.py",
#                 ])
#
#
# # pose estimation, with YOLO detection model
# # Run ViTPose on the two color images, save shoulders & hips keypoints.
# subprocess.run([
#                 "python", "ViTPose/demo/top_down_img_demo_with_mmdet.py",
#                 "ViTPose/demo/mmdetection_cfg/yolov3_d53_320_273e_coco.py",
#                 "ViTPose/models/yolov3_d53_320_273e_coco.pth",
#                 "ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py",
#                 "ViTPose/models/vitpose-l-multi-coco.pth",
#                 "--subject_name={}".format(SUBJECT_NAME),
#                 "--scan_pose={}".format(SCAN_POSE),
#                 ])
#
# # nipple detection
# # run DeepNipple on the two color images, save nipple keypoints.
# subprocess.run([
#                 "python", "DeepNipple/base_deepnipple.py",
#                 "--mode=bbox",
#                 "--show=True"
#                 ])

# Compute target coordinates, and save them
subprocess.run([
                "python", "final_phase/compute_target.py",
                "--pose={}".format(SCAN_POSE)
                ])

# # Move robot
# subprocess.run([
#                 "python", "final_phase/ViTPose_UR_move.py"
#                ])
