# Copyright (c) OpenMMLab. All rights reserved.
import os
import pickle
import warnings
import glob
from argparse import ArgumentParser
import time

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--img', type=str, default='', help='Image file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--subject_name',
        type=str,
        default="john_doe",
        help='subject name')
    parser.add_argument(
        '--scan_pose',
        type=str,
        default="front",
        help='scan_pose')
    parser.add_argument(
        '--pose_model',
        type=str,
        default="ViTPose_large",
        choices=['ViTPose_large', 'ViTPose_base'],
        help='ViTPose model')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    # assert args.show or (args.out_img_root != '')
    # assert args.img != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    start_time = time.time()
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # print("load detection model: {:.3f} s".format(time.time() - start_time))
    # build the pose model from a config file and a checkpoint file

    start_time = time.time()
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())
    # print("load pose estimation model: {:.3f} s".format(time.time() - start_time))

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    folder_path = 'final_phase/data/' + args.subject_name + '/' + args.scan_pose + '/'
    img_root = folder_path + "color_images"
    os.chdir(img_root)
    for i, file in enumerate(glob.glob("*.jpg")):
        image_name = os.path.join(file)
        # print(image_name)

        # test a single image, the resulting box is (x1, y1, x2, y2)
        start_time = time.time()
        mmdet_results = inference_detector(det_model, image_name)
        # print("detection time: {:.3f} s".format(time.time()-start_time))

        # keep the person class bounding boxes.
        start_time = time.time()
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)
        # print("pose estimation time: {:.3f} s".format(time.time() - start_time))

        # test a single image, with a list of bboxes.

        # optional
        return_heatmap = False

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # save pose_results to a file
        with open('../' + args.pose_model + '/keypoints/cam_{}_keypoints.pickle'.format(i+1), 'wb') as f:
            pickle.dump(pose_results, f)

        out_img_root = '../' + args.pose_model + '/output_images'
        os.makedirs(out_img_root, exist_ok=True)
        out_file = os.path.join(out_img_root, f'output_{i+1}.jpg')

        # show the results
        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=args.show,
            out_file=out_file)


if __name__ == '__main__':
    main()
