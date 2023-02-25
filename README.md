<h1 align="center">Localizing Scan Targets from Human Pose for Autonomous Lung Ultrasound Imaging</h1>

<h4 align="center">This is the official repository of the paper <a href="http://arxiv.org/abs/2212.07867">Localizing Scan Targets from Human Pose for Autonomous Lung Ultrasound Imaging</a>.</h4>
<h5 align="center"><em>Jianzhi Long<sup>1,2&#8727;</sup>, Jicang Cai<sup>2&#8727;</sup>, Abdullah F. Al-Battal<sup>2</sup>, Shiwei Jin<sup>2</sup>, Jing Zhang<sup>1</sup>, Dacheng Tao<sup>1,3</sup>, Imanuel Lerman<sup>2</sup>, Truong Nguyen<sup>2</sup></em></h5>
<h6 align="center">1 The University of Sydney, Australia; 2 University of California San Diego, USA; 3 JD Explore Academy, Beijing, China </h6>

<p align="center">
  <a href="#introduction">Introduction</a> |
  <a href="#scan-targets">Scan Targets</a> |
  <a href="#system-setup">System Setup</a> |
  <a href="#pipeline">Pipeline</a> |
  <a href="#running-the-code">Run Code</a> |
  <a href="#demo-video">Demo Video</a> |
  <a href="#installation">Installation</a> |
  <a href="#contact-info">Contact Info</a> |
  <a href="#acknowledge">Acknowledge</a> |
  <a href="#statement">Statement</a>
</p>

## Introduction
This repository contains the code, experiment results and a video demo for the paper Localizing Scan Targets from Human Pose for
Autonomous Lung Ultrasound Imaging. Scan target localization is defined as moving the Ultrasound (US) transducer probe to the proximity of the target scan location. We combined a human pose estimation model with a specially designed interpolation model to predict the lung ultrasound scan targets, while multi-view stereo vision is deployed to enhance the accuracy of 3D target localization.

We have released the code for [implementation](src) of our proposed [pipeline](#Pipeline) with the [system setup](#system-setup) shown below, as well as the [evaluation](src/evaluation) of the system performance. We also included a short [video demo](#demo-video) of localizing the scan target on a human subject to show the system in action.  

## Scan Targets:
<img src="homepage/target_scan_locations.png" width="60%"/>
In our project, we focus on localizing scan targets 1, 2, and 4.

## System Setup:
<img src='homepage/apparatus.png' width="60%" />

## Pipeline:
<img src='homepage/pipeline.png' width="100%" />

## Running the Code
Detailed instructions of running the code are included in other `README.md` files:
- To perform one scanning trial, see <a href="https://github.com/JamesLong199/Autonomous-Transducer-Project/tree/main/src">`src/README.md`</a>.
- To evaluate results, see <a href="https://github.com/JamesLong199/Autonomous-Transducer-Project/tree/main/src/evaluation">`src/evaluation/README.md`</a>.

## Demo Video 
https://user-images.githubusercontent.com/60713478/221347137-9e76d059-1eaa-453e-aa6f-1683a4696ee8.mp4

## Installation

1. Clone this repository

    `git clone https://github.com/JamesLong199/Autonomous-Transducer-Project.git`

2. Go into the repository

    `cd Autonomous-Transducer-Project`

3. Create conda environment and activate

    `conda create -n Auto_US python=3.7`

    `conda activate Auto_US`

4. Install dependencies

    `pip install -r requirements.txt`

5. Download [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) models and use the corresponding config file. Place the models in `ViTPose/models`. We use 
   1. ViTPose-L (COCO+AIC+MPII+CrowdPose)
   2. ViTPose-B (classic decoder, COCO)

6. Download detector model for ViTPose. Place the model in `ViTPose/models`. We use
   1. [YOLOv3](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo) (DarkNet-53, 320, 273e)

7. To use OpenPose, follow the instructions on the [official OpenPose documentations](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md) to download the Windows Portable Demo. Place the `openpose` folder in the project's root directory.

Our code has been tested with Python 3.7.11 on Windows 11.

## Contact Info
| Name  | Email |
| ------------- | ------------- |
| Jianzhi Long  | jlong@ucsd.edu |
| Jicang Cai  | j1cai@ucsd.edu  |

## Acknowledge
We acknowledge the excellent implementation from [ViTPose](https://github.com/ViTAE-Transformer/ViTPose), [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), and Rope Robotics (Denmark).

## Statement
Will become available once the paper is on arxiv.
