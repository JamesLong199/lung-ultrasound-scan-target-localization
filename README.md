# Localizing Scan Targets from Human Pose for Autonomous Lung Ultrasound Imaging

(Under construction...)

## Introduction
This repository contains the code, experiment results and a video demo for the paper Localizing Scan Targets from Human Pose for
Autonomous Lung Ultrasound Imaging. Scan target localization is defined as moving the Ultrasound (US) transducer probe to the proximity of the target scan location. We combined a human pose estimation model with a specially designed interpolation model to predict the lung ultrasound scan targets, while multi-view stereo vision is deployed to enhance the accuracy of 3D target localization.

We have released the code for [implementation](src) of our proposed [pipeline](#Pipeline) with the [system setup](#SystemSetup) shown below, as well as the [evaluation](src/evaluation) of the system performance. We also included a short [video demo](#DemoVideo) of localizing the scan target on a human subject to show the system in action.  

## 9 scan target locations (we focus on target 1,2,4):
<img src="homepage/target_scan_locations.png" width="450"/>

## System Setup <div id="SystemSetup"></div>
<img src='homepage/apparatus.png' width="450" />

## Pipeline <div id="Pipeline"></div>
<img src='homepage/pipeline.png' width="800" />

## Demo Video <div id="DemoVideo"></div>
https://user-images.githubusercontent.com/66498825/187047342-1848f07d-ceaf-44e0-8098-28f5038e718b.mp4

## Installation
- URBasic 
- ViTPose
- OpenPose

## Contributors
| Name  | Email |
| ------------- | ------------- |
| Jianzhi Long  | jlong@ucsd.edu |
| Jicang Cai  | j1cai@ucsd.edu  |

## Acknowledge
We acknowledge the excellent implementation from [ViTPose](https://github.com/ViTAE-Transformer/ViTPose), [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), and Rope Robotics (Denmark).

